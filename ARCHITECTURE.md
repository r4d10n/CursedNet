# CursedNet: Deep Learning Blind FM Demodulation

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Blind Demodulation Theory](#blind-demodulation-theory)
4. [Training Data Generation](#training-data-generation)
5. [Training Pipeline](#training-pipeline)
6. [Inference](#inference)
7. [Mathematical Foundation](#mathematical-foundation)

---

## Overview

CursedNet is a 1D convolutional neural network that learns to perform **blind FM (Frequency Modulation) demodulation** directly from raw RF (Radio Frequency) signals. Unlike traditional FM demodulators that use explicit mathematical operations (differentiation, phase detection), CursedNet learns the demodulation function end-to-end from paired training examples.

### Key Characteristics
- **Input**: Complex RF signal (I/Q samples) at 256 kHz sample rate
- **Output**: Demodulated mono audio at 32 kHz sample rate
- **Downsampling Factor**: 8x (achieved through max-pooling)
- **Architecture Style**: Encoder-only CNN inspired by U-Net principles

---

## Model Architecture

### High-Level Structure

```
                    CursedNet Architecture
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   Input: [batch, 2, 256000]  (I/Q complex signal)   │
    │                      │                              │
    │              ┌───────▼───────┐                      │
    │              │   Conv1 (32)  │  conv_block          │
    │              │   Conv2 (64)  │  conv_block          │
    │              │   Conv3 (64)  │  conv_block          │
    │              └───────┬───────┘                      │
    │                      │                              │
    │              ┌───────▼───────┐                      │
    │              │  MaxPool (2)  │  Downsample 2x       │
    │              └───────┬───────┘                      │
    │                      │                              │
    │              ┌───────▼───────┐                      │
    │              │   Conv4 (32)  │  conv_block          │
    │              │   Conv5 (16)  │  conv_block          │
    │              └───────┬───────┘                      │
    │                      │                              │
    │              ┌───────▼───────┐                      │
    │              │  MaxPool (2)  │  Downsample 2x       │
    │              └───────┬───────┘                      │
    │                      │                              │
    │              ┌───────▼───────┐                      │
    │              │   Conv6 (8)   │  conv_block          │
    │              │   Conv7 (4)   │  conv_block          │
    │              └───────┬───────┘                      │
    │                      │                              │
    │              ┌───────▼───────┐                      │
    │              │  MaxPool (2)  │  Downsample 2x       │
    │              └───────┬───────┘                      │
    │                      │                              │
    │              ┌───────▼───────┐                      │
    │              │  Conv8 (1x1)  │  Channel reduction   │
    │              │     Tanh      │  Output activation   │
    │              └───────┬───────┘                      │
    │                      │                              │
    │   Output: [batch, 1, 32000]  (Demodulated audio)    │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

### Detailed Layer Specifications

#### conv_block Module
Each `conv_block` contains a dual-convolution structure:

```python
conv_block(in_ch, out_ch):
    Conv1d(in_ch, out_ch, kernel=3, stride=1, padding=1)
    BatchNorm1d(out_ch)
    Tanh()
    Conv1d(out_ch, out_ch, kernel=3, stride=1, padding=1)
    BatchNorm1d(out_ch)
    Tanh()
```

**Design Rationale**:
- **Kernel size 3**: Captures local frequency patterns (±1 sample context)
- **Padding 1**: Preserves temporal dimension through convolutions
- **BatchNorm**: Stabilizes training, enables higher learning rates
- **Tanh activation**: Bounded output [-1, 1], matches audio signal range

#### Layer-by-Layer Breakdown

| Layer | Input Channels | Output Channels | Spatial Dim | Operation |
|-------|---------------|-----------------|-------------|-----------|
| Conv1 | 2 (I/Q) | 32 | 256000 | Feature extraction |
| Conv2 | 32 | 64 | 256000 | Feature expansion |
| Conv3 | 64 | 64 | 256000 | Feature refinement |
| MaxPool1 | 64 | 64 | 128000 | 2x downsampling |
| Conv4 | 64 | 32 | 128000 | Channel reduction |
| Conv5 | 32 | 16 | 128000 | Channel reduction |
| MaxPool2 | 16 | 16 | 64000 | 2x downsampling |
| Conv6 | 16 | 8 | 64000 | Channel reduction |
| Conv7 | 8 | 4 | 64000 | Channel reduction |
| MaxPool3 | 4 | 4 | 32000 | 2x downsampling |
| Conv8 | 4 | 1 | 32000 | Output projection |
| Tanh | 1 | 1 | 32000 | Output normalization |

**Total Downsampling**: 2 × 2 × 2 = **8x** (256 kHz → 32 kHz)

### Parameter Count

```
Conv1:  2×32×3×2 + 32×32×3×2 = 384 + 6144 = 6,528
Conv2:  32×64×3×2 + 64×64×3×2 = 12,288 + 24,576 = 36,864
Conv3:  64×64×3×2 × 2 = 49,152
Conv4:  64×32×3×2 + 32×32×3×2 = 12,288 + 6,144 = 18,432
Conv5:  32×16×3×2 + 16×16×3×2 = 3,072 + 1,536 = 4,608
Conv6:  16×8×3×2 + 8×8×3×2 = 768 + 384 = 1,152
Conv7:  8×4×3×2 + 4×4×3×2 = 192 + 96 = 288
Conv8:  4×1×1 = 4
BatchNorm params: ~500

Total: ~117,500 trainable parameters
```

---

## Blind Demodulation Theory

### What is Blind Demodulation?

**Blind demodulation** refers to recovering the original message signal from a modulated carrier without explicit knowledge of:
- Carrier frequency (learned implicitly)
- Modulation index
- Demodulation algorithm

The neural network learns these implicitly through supervised training.

### Traditional FM Demodulation vs Neural Approach

#### Traditional Algorithm (Reference Implementation)

FM demodulation traditionally involves:

1. **Frequency Discrimination**: Detect instantaneous frequency
   ```
   f(t) = (1/2π) × d(phase)/dt
   ```

2. **Phase Differentiation**:
   ```
   audio(t) = d/dt[arctan(Q(t)/I(t))]
   ```

3. **De-emphasis Filtering**: Apply low-pass filter with time constant τ (75μs for US FM)

4. **Decimation**: Downsample from RF rate to audio rate

#### Neural Network Approach

CursedNet learns an **implicit approximation** of these operations:

1. **Input Representation**:
   - Complex signal split into Real (I) and Imaginary (Q) channels
   - `input = [Re(signal), Im(signal)]` → Shape: [2, 256000]

2. **Learned Frequency Detection**:
   - First convolution layers learn local phase/frequency patterns
   - Kernel size 3 provides ±1 sample context for derivative estimation
   - Network implicitly learns: `Δphase ≈ atan2(Q[n], I[n]) - atan2(Q[n-1], I[n-1])`

3. **Learned Filtering**:
   - Deeper layers learn frequency-selective filtering
   - BatchNorm provides automatic gain control
   - Multiple conv layers approximate de-emphasis response

4. **Learned Decimation**:
   - MaxPool layers perform anti-aliased downsampling
   - Network learns what information to preserve during decimation

### Why This Works: Universal Approximation

Convolutional neural networks are **universal function approximators** for translation-equivariant functions. FM demodulation is fundamentally:
- A local operation (depends on nearby samples)
- Translation equivariant (shifting input shifts output)
- Continuous and differentiable

This makes it well-suited for CNN approximation.

### Advantages of Neural Demodulation

| Aspect | Traditional | Neural |
|--------|-------------|--------|
| Adaptation | Fixed parameters | Learned from data |
| Robustness | Sensitive to noise | Can learn noise rejection |
| Flexibility | Single modulation scheme | Potentially multi-scheme |
| Implementation | DSP expertise required | Data-driven |
| Latency | Low (streaming) | Batch processing |

---

## Training Data Generation

Training data consists of **paired examples**: raw RF signal (input) and corresponding demodulated audio (output/label).

### Method 1: Hardware Collection (Current Implementation)

Using Software Defined Radio (SDR) hardware to capture real-world FM broadcasts:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Antenna   │────▶│  SDR Radio  │────▶│   Computer  │
│  (FM Band)  │     │ (AirspyHF)  │     │  (collect)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Complex IF Signal      │
              │ (256 kHz, Complex64)   │
              └───────────┬────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   ┌─────────────────┐         ┌─────────────────┐
   │  Save as Input  │         │ MFM Demodulator │
   │  IF_{n}.npy     │         │ (Reference Algo)│
   └─────────────────┘         └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Save as Label  │
                               │  FM_{n}.npy     │
                               └─────────────────┘
```

#### Collection Script Details (`collect.py`)

```python
# SDR Configuration
freq = 96.9e6        # FM station frequency (96.9 MHz)
sfs = int(256e3)     # RF sample rate (256 kHz)
afs = int(32e3)      # Audio sample rate (32 kHz)
tau = 75e-6          # De-emphasis time constant (US standard)

# Reference demodulator
demod = MFM(tau, sfs, afs, cuda=False)

# For each 1-second chunk:
# 1. Capture 256000 complex samples from SDR
# 2. Save as IF_{counter}.npy (Complex64)
# 3. Demodulate using reference MFM
# 4. Save as FM_{counter}.npy (Float32)
```

#### Hardware Requirements
- **SDR**: AirspyHF (or compatible SoapySDR device)
- **Antenna**: FM band antenna (88-108 MHz)
- **Computer**: CUDA GPU optional for accelerated reference demod

### Method 2: Synthetic Data Generation (Recommended for Training)

Generate unlimited training data without hardware by simulating the FM modulation process:

```python
import numpy as np
from scipy.signal import resample

def generate_fm_training_pair(
    audio_source,           # Source audio file or synthetic
    carrier_freq=0,         # Baseband (0 for complex baseband)
    sample_rate_rf=256000,  # RF sample rate
    sample_rate_audio=32000,# Audio sample rate
    freq_deviation=75000,   # FM deviation (±75 kHz for broadcast)
    duration=1.0,           # Duration in seconds
    snr_db=30               # Signal-to-noise ratio
):
    """
    Generate a synthetic FM modulated signal and its audio pair.
    """
    n_samples_rf = int(sample_rate_rf * duration)
    n_samples_audio = int(sample_rate_audio * duration)

    # Step 1: Get or generate audio signal
    if audio_source is None:
        # Generate synthetic audio (speech-like or music-like)
        t_audio = np.linspace(0, duration, n_samples_audio)
        # Multi-tone to simulate speech spectrum
        audio = np.zeros(n_samples_audio)
        for f in [200, 400, 800, 1200, 2000, 3000]:
            audio += np.random.uniform(0.1, 1.0) * np.sin(2*np.pi*f*t_audio + np.random.uniform(0, 2*np.pi))
        audio = audio / np.max(np.abs(audio)) * 0.8
    else:
        audio = load_and_resample(audio_source, sample_rate_audio, duration)

    # Step 2: Upsample audio to RF rate
    audio_upsampled = resample(audio, n_samples_rf)

    # Step 3: FM Modulation
    # Instantaneous frequency: f(t) = fc + k_f * m(t)
    # Phase: φ(t) = 2π * fc * t + 2π * k_f * ∫m(τ)dτ

    k_f = freq_deviation  # Frequency sensitivity
    t_rf = np.linspace(0, duration, n_samples_rf)

    # Integrate audio to get phase
    phase = 2 * np.pi * k_f * np.cumsum(audio_upsampled) / sample_rate_rf

    # Generate complex baseband signal
    signal = np.exp(1j * phase)

    # Step 4: Add noise
    if snr_db < np.inf:
        noise_power = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(n_samples_rf) + 1j * np.random.randn(n_samples_rf))
        signal = signal + noise

    # Step 5: Prepare training pair
    # Input: Complex signal as 2-channel real (I, Q)
    input_signal = np.stack([np.real(signal), np.imag(signal)]).astype(np.float32)

    # Output: Original audio (normalized)
    output_audio = (audio / np.max(np.abs(audio))).astype(np.float32)

    return input_signal, output_audio
```

#### Synthetic Data Variations

To create robust training data, vary these parameters:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| SNR | 10-40 dB | Noise robustness |
| Frequency deviation | 50-100 kHz | Handle different stations |
| Audio content | Speech, music, silence | Content diversity |
| Multipath | 0-3 echoes | Channel robustness |
| Doppler | ±10 Hz | Motion robustness |
| DC offset | ±0.1 | Hardware imperfections |
| I/Q imbalance | ±5% gain, ±5° phase | SDR imperfections |

#### Complete Synthetic Dataset Generator

```python
import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import resample
import glob

class FMDatasetGenerator:
    def __init__(self,
                 output_dir,
                 rf_sample_rate=256000,
                 audio_sample_rate=32000,
                 chunk_duration=1.0):
        self.output_dir = output_dir
        self.rf_sr = rf_sample_rate
        self.audio_sr = audio_sample_rate
        self.chunk_duration = chunk_duration
        os.makedirs(output_dir, exist_ok=True)

    def generate_from_audio_files(self, audio_dir, num_augmentations=5):
        """Generate dataset from directory of audio files."""
        audio_files = glob.glob(f"{audio_dir}/*.wav")
        counter = 0

        for audio_file in audio_files:
            sr, audio = wavfile.read(audio_file)
            audio = audio.astype(np.float32) / 32768.0

            # Resample to target audio rate
            if sr != self.audio_sr:
                num_samples = int(len(audio) * self.audio_sr / sr)
                audio = resample(audio, num_samples)

            # Process in chunks
            chunk_samples = int(self.audio_sr * self.chunk_duration)
            num_chunks = len(audio) // chunk_samples

            for i in range(num_chunks):
                chunk = audio[i*chunk_samples:(i+1)*chunk_samples]

                # Generate augmented versions
                for aug in range(num_augmentations):
                    snr = np.random.uniform(15, 40)
                    deviation = np.random.uniform(60000, 90000)

                    rf_signal = self._modulate_fm(chunk, deviation, snr)

                    # Save pair
                    np.save(f"{self.output_dir}/IF_{counter}.npy", rf_signal)
                    np.save(f"{self.output_dir}/FM_{counter}.npy", chunk)
                    counter += 1

        print(f"Generated {counter} training pairs")

    def _modulate_fm(self, audio, freq_deviation, snr_db):
        """FM modulate audio signal."""
        # Upsample
        n_rf = int(len(audio) * self.rf_sr / self.audio_sr)
        audio_up = resample(audio, n_rf)

        # Modulate
        phase = 2 * np.pi * freq_deviation * np.cumsum(audio_up) / self.rf_sr
        signal = np.exp(1j * phase)

        # Add noise
        noise_power = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(n_rf) + 1j * np.random.randn(n_rf)
        )
        signal = signal + noise

        return signal.astype(np.complex64)

    def generate_synthetic(self, num_samples):
        """Generate purely synthetic training data."""
        for i in range(num_samples):
            # Random audio synthesis
            t = np.linspace(0, self.chunk_duration, int(self.audio_sr * self.chunk_duration))

            # Randomly choose content type
            content_type = np.random.choice(['tones', 'noise', 'sweep', 'silence'])

            if content_type == 'tones':
                audio = self._generate_tones(t)
            elif content_type == 'noise':
                audio = self._generate_filtered_noise(t)
            elif content_type == 'sweep':
                audio = self._generate_sweep(t)
            else:
                audio = np.zeros_like(t)

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8

            # Modulate with random parameters
            snr = np.random.uniform(10, 45)
            deviation = np.random.uniform(50000, 100000)

            rf_signal = self._modulate_fm(audio.astype(np.float32), deviation, snr)

            np.save(f"{self.output_dir}/IF_{i}.npy", rf_signal)
            np.save(f"{self.output_dir}/FM_{i}.npy", audio.astype(np.float32))

    def _generate_tones(self, t):
        """Generate multi-tone signal."""
        audio = np.zeros_like(t)
        num_tones = np.random.randint(3, 10)
        for _ in range(num_tones):
            freq = np.random.uniform(100, 4000)
            amp = np.random.uniform(0.1, 1.0)
            phase = np.random.uniform(0, 2*np.pi)
            audio += amp * np.sin(2*np.pi*freq*t + phase)
        return audio

    def _generate_filtered_noise(self, t):
        """Generate bandlimited noise."""
        from scipy.signal import butter, filtfilt
        noise = np.random.randn(len(t))
        # Bandpass 100-4000 Hz
        b, a = butter(4, [100, 4000], btype='band', fs=self.audio_sr)
        return filtfilt(b, a, noise)

    def _generate_sweep(self, t):
        """Generate frequency sweep."""
        f0 = np.random.uniform(100, 500)
        f1 = np.random.uniform(1000, 4000)
        return np.sin(2*np.pi * (f0 + (f1-f0)*t/t[-1]) * t)
```

### Method 3: Using Public Audio Datasets

For speech-focused training, use public datasets:

| Dataset | Content | Size | License |
|---------|---------|------|---------|
| LibriSpeech | Audiobooks | 1000 hrs | CC BY 4.0 |
| Common Voice | Read speech | 10000+ hrs | CC0 |
| VoxCeleb | Celebrity speech | 2000+ hrs | CC BY-SA 4.0 |
| FMA | Music | 1000 hrs | Various |
| ESC-50 | Environmental | 40 hrs | CC BY-NC |

```python
# Example: Generate dataset from LibriSpeech
generator = FMDatasetGenerator(output_dir="dataset")
generator.generate_from_audio_files(
    audio_dir="/path/to/librispeech/dev-clean",
    num_augmentations=10  # 10 SNR/deviation variations per clip
)
```

---

## Training Pipeline

### Dataset Class Implementation

The training requires a `MultiChannelDataset` class (missing from repo):

```python
# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiChannelDataset(Dataset):
    """
    Dataset for paired IF/FM numpy files.

    Expected directory structure:
        root_dir/
            IF_0.npy    # Complex64, shape (256000,)
            FM_0.npy    # Float32, shape (32000,)
            IF_1.npy
            FM_1.npy
            ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Find all IF files and extract indices
        self.indices = []
        for f in os.listdir(root_dir):
            if f.startswith('IF_') and f.endswith('.npy'):
                idx = int(f[3:-4])  # Extract number
                fm_file = f"FM_{idx}.npy"
                if os.path.exists(os.path.join(root_dir, fm_file)):
                    self.indices.append(idx)

        self.indices.sort()
        print(f"Found {len(self.indices)} paired samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]

        # Load input (IF signal)
        if_path = os.path.join(self.root_dir, f"IF_{sample_idx}.npy")
        if_data = np.load(if_path)  # Complex64

        # Convert complex to 2-channel real
        # Shape: (2, 256000) for [Real, Imaginary]
        input_data = np.stack([
            np.real(if_data),
            np.imag(if_data)
        ]).astype(np.float32)

        # Load output (FM audio)
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
```

### Training Configuration

```python
# main.py configuration
root_dir = "dataset"           # Directory containing IF_*.npy and FM_*.npy
learning_rate = 0.01           # Initial learning rate
min_lr = 1e-6                  # Early stopping threshold
epochs = 500                   # Maximum epochs
batch_size = 2                 # Samples per batch (memory limited)
percentages = [0.85, 0.15, 0]  # Train/Val/Test split
patience = 10                  # LR scheduler patience
```

### Loss Function

**Mean Squared Error (MSE)** is used:

```
L = (1/N) Σ (y_pred - y_true)²
```

This is appropriate because:
- Audio is continuous valued
- Penalizes large errors more than small ones
- Gradient is smooth and stable

Alternative losses to consider:
- **L1 Loss**: More robust to outliers
- **Perceptual Loss**: Uses pre-trained audio features
- **Multi-resolution STFT Loss**: Captures spectral accuracy

### Optimizer

**Adam** with modified betas:
- β₁ = 0.5 (reduced from default 0.9)
- β₂ = 0.999 (default)

Lower β₁ provides faster adaptation, useful for time-series data.

### Learning Rate Schedule

**ReduceLROnPlateau**:
- Monitors validation loss
- Reduces LR by factor of 0.1 when loss plateaus
- Patience parameter controls sensitivity

---

## Inference

### Real-time Pipeline (`player.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                    Real-time Inference                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐    ┌─────────────┐    ┌──────────────────┐   │
│   │   SDR   │───▶│ Complex→I/Q │───▶│    CursedNet     │   │
│   │ 256 kHz │    │  [2, 256k]  │    │ Forward Pass     │   │
│   └─────────┘    └─────────────┘    └────────┬─────────┘   │
│                                               │             │
│                                               ▼             │
│                                      ┌──────────────────┐   │
│                                      │  Audio Output    │   │
│                                      │   32 kHz Mono    │   │
│                                      └──────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Inference Code

```python
# Process single chunk
def neural_demodulate(model, complex_signal, device):
    """
    Demodulate using trained CursedNet.

    Args:
        model: Trained CursedNet model
        complex_signal: np.array of complex64, shape (256000,)
        device: torch device

    Returns:
        audio: np.array of float32, shape (32000,)
    """
    # Convert complex to I/Q channels
    iq = np.stack([np.real(complex_signal), np.imag(complex_signal)])

    # Add batch dimension
    iq = np.expand_dims(iq, axis=0).astype(np.float32)

    # Inference
    with torch.no_grad():
        input_tensor = torch.tensor(iq).to(device)
        output = model(input_tensor)

    # Extract audio
    audio = output[0, 0].cpu().numpy()

    return audio
```

---

## Mathematical Foundation

### FM Modulation Mathematics

Given a message signal m(t) (audio), FM modulation produces:

```
s(t) = A·cos(2πf_c·t + 2πk_f·∫m(τ)dτ)
```

Where:
- A = carrier amplitude
- f_c = carrier frequency
- k_f = frequency deviation constant

In complex baseband representation:

```
s(t) = A·exp(j·2πk_f·∫m(τ)dτ)
```

### FM Demodulation Mathematics

To recover m(t), differentiate the phase:

```
φ(t) = 2πk_f·∫m(τ)dτ

m(t) = (1/2πk_f)·dφ/dt
```

The phase can be extracted from I/Q:

```
φ(t) = arctan(Q(t)/I(t))
```

### Neural Network Approximation

The network learns an approximation:

```
m̂(t) ≈ f_θ(I(t), Q(t))
```

Where f_θ is the learned function parameterized by network weights θ.

Through training, the network learns to implicitly compute:
1. Phase extraction: arctan(Q/I)
2. Differentiation: via convolutional kernels
3. Filtering: de-emphasis and anti-aliasing
4. Decimation: via max-pooling

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `model.py:1-62` | 62 | CursedNet architecture definition |
| `train.py:1-161` | 161 | Training loop and checkpointing |
| `eval.py:1-28` | 28 | Validation evaluation |
| `collect.py:1-73` | 73 | SDR data collection |
| `player.py:1-68` | 68 | Real-time inference |
| `main.py:1-22` | 22 | Training configuration |
| `utils.py:1-3` | 3 | Learning rate helper |

---

## References

1. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
2. **FM Demodulation**: Carlson, "Communication Systems" (4th ed.)
3. **Neural Audio Processing**: Engel et al., "DDSP: Differentiable Digital Signal Processing" (2020)
