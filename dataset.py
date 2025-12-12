"""
MultiChannelDataset for CursedNet training.

This module provides the dataset class for loading paired IF/FM numpy files
for training the blind FM demodulation neural network.

Expected directory structure:
    root_dir/
        IF_0.npy    # Complex64, shape (256000,) - Raw RF signal
        FM_0.npy    # Float32, shape (32000,)   - Demodulated audio
        IF_1.npy
        FM_1.npy
        ...
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiChannelDataset(Dataset):
    """
    Dataset for paired IF (input) and FM (output) numpy files.

    The input IF files contain complex baseband RF signals captured at 256 kHz.
    The output FM files contain the reference demodulated audio at 32 kHz.

    Args:
        root_dir (str): Path to directory containing IF_*.npy and FM_*.npy files
        transform (callable, optional): Optional transform to apply to samples
        input_channels (int): Number of input channels (1 for magnitude, 2 for I/Q)
    """

    def __init__(self, root_dir, transform=None, input_channels=2):
        self.root_dir = root_dir
        self.transform = transform
        self.input_channels = input_channels

        # Find all IF files and extract indices
        self.indices = []
        for f in os.listdir(root_dir):
            if f.startswith('IF_') and f.endswith('.npy'):
                try:
                    idx = int(f[3:-4])  # Extract number from IF_{idx}.npy
                    fm_file = f"FM_{idx}.npy"
                    if os.path.exists(os.path.join(root_dir, fm_file)):
                        self.indices.append(idx)
                except ValueError:
                    continue  # Skip files that don't match pattern

        self.indices.sort()
        print(f"MultiChannelDataset: Found {len(self.indices)} paired samples in {root_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Load and return a single training sample.

        Args:
            idx (int): Index of sample to load

        Returns:
            dict: Dictionary with 'input' and 'output' tensors
                - input: torch.Tensor of shape (input_channels, 256000)
                - output: torch.Tensor of shape (1, 32000)
        """
        sample_idx = self.indices[idx]

        # Load input (IF signal - complex baseband)
        if_path = os.path.join(self.root_dir, f"IF_{sample_idx}.npy")
        if_data = np.load(if_path)  # Complex64, shape (256000,)

        # Convert complex to real channels based on input_channels setting
        if self.input_channels == 2:
            # Use I/Q representation: [Real, Imaginary]
            input_data = np.stack([
                np.real(if_data),
                np.imag(if_data)
            ]).astype(np.float32)
        elif self.input_channels == 1:
            # Use magnitude only
            input_data = np.abs(if_data).astype(np.float32)
            input_data = input_data.reshape(1, -1)
        else:
            raise ValueError(f"input_channels must be 1 or 2, got {self.input_channels}")

        # Load output (FM audio - demodulated reference)
        fm_path = os.path.join(self.root_dir, f"FM_{sample_idx}.npy")
        output_data = np.load(fm_path).astype(np.float32)

        # Ensure correct shape: (1, 32000)
        if output_data.ndim == 1:
            output_data = output_data.reshape(1, -1)

        sample = {
            'input': torch.from_numpy(input_data),
            'output': torch.from_numpy(output_data)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class SyntheticFMDataset(Dataset):
    """
    On-the-fly synthetic FM dataset generation for training without hardware.

    Generates FM modulated signals from synthetic audio sources with
    configurable SNR, frequency deviation, and audio characteristics.

    Args:
        num_samples (int): Number of samples to generate per epoch
        rf_sample_rate (int): RF sample rate (default: 256000)
        audio_sample_rate (int): Audio sample rate (default: 32000)
        duration (float): Duration of each sample in seconds (default: 1.0)
        snr_range (tuple): Range of SNR values in dB (default: (15, 40))
        deviation_range (tuple): Range of freq deviation in Hz (default: (60000, 90000))
        seed (int, optional): Random seed for reproducibility
    """

    def __init__(self,
                 num_samples,
                 rf_sample_rate=256000,
                 audio_sample_rate=32000,
                 duration=1.0,
                 snr_range=(15, 40),
                 deviation_range=(60000, 90000),
                 seed=None):
        self.num_samples = num_samples
        self.rf_sr = rf_sample_rate
        self.audio_sr = audio_sample_rate
        self.duration = duration
        self.snr_range = snr_range
        self.deviation_range = deviation_range

        if seed is not None:
            np.random.seed(seed)

        self.n_rf = int(rf_sample_rate * duration)
        self.n_audio = int(audio_sample_rate * duration)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a synthetic FM training sample."""
        # Generate random audio
        audio = self._generate_audio()

        # Random modulation parameters
        snr = np.random.uniform(*self.snr_range)
        deviation = np.random.uniform(*self.deviation_range)

        # FM modulate
        rf_signal = self._fm_modulate(audio, deviation, snr)

        # Prepare tensors
        input_data = np.stack([
            np.real(rf_signal),
            np.imag(rf_signal)
        ]).astype(np.float32)

        output_data = audio.reshape(1, -1).astype(np.float32)

        return {
            'input': torch.from_numpy(input_data),
            'output': torch.from_numpy(output_data)
        }

    def _generate_audio(self):
        """Generate synthetic audio signal."""
        from scipy.signal import resample

        t = np.linspace(0, self.duration, self.n_audio)
        content_type = np.random.choice(['tones', 'noise', 'sweep', 'silence'], p=[0.4, 0.3, 0.2, 0.1])

        if content_type == 'tones':
            audio = self._generate_tones(t)
        elif content_type == 'noise':
            audio = self._generate_filtered_noise()
        elif content_type == 'sweep':
            audio = self._generate_sweep(t)
        else:
            audio = np.zeros(self.n_audio)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio.astype(np.float32)

    def _generate_tones(self, t):
        """Generate multi-tone signal simulating speech/music."""
        audio = np.zeros(len(t))
        num_tones = np.random.randint(3, 12)
        for _ in range(num_tones):
            freq = np.random.uniform(100, 4000)
            amp = np.random.uniform(0.1, 1.0)
            phase = np.random.uniform(0, 2*np.pi)
            audio += amp * np.sin(2*np.pi*freq*t + phase)
        return audio

    def _generate_filtered_noise(self):
        """Generate bandlimited noise in speech frequency range."""
        from scipy.signal import butter, filtfilt
        noise = np.random.randn(self.n_audio)
        # Bandpass 100-4000 Hz (speech range)
        b, a = butter(4, [100/(self.audio_sr/2), 4000/(self.audio_sr/2)], btype='band')
        return filtfilt(b, a, noise)

    def _generate_sweep(self, t):
        """Generate frequency sweep signal."""
        f0 = np.random.uniform(100, 500)
        f1 = np.random.uniform(1000, 4000)
        return np.sin(2*np.pi * (f0 + (f1-f0)*t/t[-1]) * t)

    def _fm_modulate(self, audio, freq_deviation, snr_db):
        """FM modulate audio signal."""
        from scipy.signal import resample

        # Upsample audio to RF rate
        audio_up = resample(audio, self.n_rf)

        # FM modulation: s(t) = exp(j * 2π * k_f * ∫m(τ)dτ)
        phase = 2 * np.pi * freq_deviation * np.cumsum(audio_up) / self.rf_sr
        signal = np.exp(1j * phase)

        # Add AWGN noise
        noise_power = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(self.n_rf) + 1j * np.random.randn(self.n_rf)
        )
        signal = signal + noise

        return signal.astype(np.complex64)


if __name__ == "__main__":
    # Test the dataset classes
    import sys

    if len(sys.argv) > 1:
        # Test with real data
        dataset = MultiChannelDataset(sys.argv[1])
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Input shape: {sample['input'].shape}")
            print(f"Output shape: {sample['output'].shape}")
            print(f"Input dtype: {sample['input'].dtype}")
            print(f"Output dtype: {sample['output'].dtype}")
    else:
        # Test synthetic dataset
        print("Testing SyntheticFMDataset...")
        dataset = SyntheticFMDataset(num_samples=10, seed=42)
        sample = dataset[0]
        print(f"Input shape: {sample['input'].shape}")
        print(f"Output shape: {sample['output'].shape}")
        print(f"Input range: [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")
        print(f"Output range: [{sample['output'].min():.3f}, {sample['output'].max():.3f}]")
        print("Synthetic dataset test passed!")
