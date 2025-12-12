# CursedNet: Deep Learning Blind FM Demodulation

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Blind Demodulation Theory](#blind-demodulation-theory)
4. [Training Data Generation](#training-data-generation)
5. [Training Pipeline](#training-pipeline)
6. [Inference](#inference)
7. [Mathematical Foundation](#mathematical-foundation)
8. [Computational Comparison: Neural vs Traditional](#computational-comparison-neural-vs-traditional-demodulation)

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

## Computational Comparison: Neural vs Traditional Demodulation

This section provides a detailed computational analysis comparing CursedNet with a traditional FM demodulation chain.

### Traditional FM Demodulation Chain

A typical software FM demodulator consists of these stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Traditional FM Demodulation Chain                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: Complex I/Q @ 256 kHz                                           │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 1. Phase Detect │  atan2(Q[n], I[n])                                 │
│  │    (CORDIC/LUT) │  Per-sample operation                              │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 2. Differentiate│  φ[n] - φ[n-1] (with unwrapping)                   │
│  │    (FIR/IIR)    │  Or FIR differentiator                             │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 3. De-emphasis  │  1st order IIR: y[n] = αx[n] + (1-α)y[n-1]        │
│  │    (IIR Filter) │  τ = 75μs (US) or 50μs (EU)                        │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 4. Anti-alias   │  Low-pass FIR filter                               │
│  │    (FIR Filter) │  Cutoff ~15 kHz, typically 64-128 taps             │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ 5. Decimate     │  Keep every 8th sample                             │
│  │    (8:1)        │  256 kHz → 32 kHz                                  │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  Output: Audio @ 32 kHz                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Operation Count Analysis

#### Traditional Chain (per 1-second chunk: 256,000 input samples → 32,000 output samples)

| Stage | Operation | Count per Sample | Total Operations | Notes |
|-------|-----------|------------------|------------------|-------|
| **1. Phase Detection** | atan2(Q, I) | ~20-60 ops | 5.1-15.4M | CORDIC: 20 iter × 3 ops; LUT: ~10 ops |
| **2. Differentiation** | Subtract + unwrap | ~5 ops | 1.28M | Simple: φ[n]-φ[n-1], unwrap ~3 ops |
| **3. De-emphasis** | 1st order IIR | 3 ops | 0.77M | 1 mul, 1 mul, 1 add |
| **4. Anti-alias Filter** | FIR convolution | 2×N taps | 32.8M | N=64 taps: 128 MACs × 256k |
| **5. Decimation** | Index selection | 0 | ~0 | Just pointer arithmetic |
| | | | | |
| **Total** | | | **~40-50M ops** | |

**Detailed FIR Calculation:**
- Anti-alias FIR with 64 taps at 256 kHz
- Operations: 64 multiplies + 63 adds = 127 ops per output
- But with polyphase decimation: only 64/8 = 8 taps per output phase
- Optimized: 8 × 2 × 32,000 = 512,000 MACs
- Naive: 127 × 256,000 = 32.5M ops

#### CursedNet Neural Network (per 1-second chunk)

**FLOPs Calculation for Each Layer:**

For Conv1d: FLOPs = 2 × K × C_in × C_out × L_out (multiply-accumulates × 2)

| Layer | Kernel | C_in | C_out | L_out | FLOPs | Cumulative |
|-------|--------|------|-------|-------|-------|------------|
| **Conv1a** | 3 | 2 | 32 | 256,000 | 98.3M | 98.3M |
| **Conv1b** | 3 | 32 | 32 | 256,000 | 1,573M | 1,671M |
| **Conv2a** | 3 | 32 | 64 | 256,000 | 3,146M | 4,817M |
| **Conv2b** | 3 | 64 | 64 | 256,000 | 6,291M | 11,108M |
| **Conv3a** | 3 | 64 | 64 | 256,000 | 6,291M | 17,399M |
| **Conv3b** | 3 | 64 | 64 | 256,000 | 6,291M | 23,690M |
| MaxPool1 | 2 | 64 | 64 | 128,000 | 8.2M | 23,698M |
| **Conv4a** | 3 | 64 | 32 | 128,000 | 1,573M | 25,271M |
| **Conv4b** | 3 | 32 | 32 | 128,000 | 786M | 26,057M |
| **Conv5a** | 3 | 32 | 16 | 128,000 | 393M | 26,450M |
| **Conv5b** | 3 | 16 | 16 | 128,000 | 197M | 26,647M |
| MaxPool2 | 2 | 16 | 16 | 64,000 | 1.0M | 26,648M |
| **Conv6a** | 3 | 16 | 8 | 64,000 | 49.2M | 26,697M |
| **Conv6b** | 3 | 8 | 8 | 64,000 | 24.6M | 26,722M |
| **Conv7a** | 3 | 8 | 4 | 64,000 | 12.3M | 26,734M |
| **Conv7b** | 3 | 4 | 4 | 64,000 | 6.1M | 26,740M |
| MaxPool3 | 2 | 4 | 4 | 32,000 | 0.1M | 26,740M |
| **Conv8** | 1 | 4 | 1 | 32,000 | 0.26M | 26,741M |
| | | | | | | |
| **Total Conv** | | | | | **~26.7 GFLOPs** | |

**Additional Operations:**

| Operation | Per Element | Elements | Total |
|-----------|-------------|----------|-------|
| BatchNorm (14×) | 4 ops | ~1.5M total | 6M |
| Tanh (15×) | ~10 ops | ~1.5M total | 15M |
| | | | **~21M** |

**Total CursedNet: ~26.7 GFLOPs**

### Comparison Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Computational Comparison (1-second chunk)                   │
├───────────────────────┬─────────────────────┬───────────────────────────┤
│        Metric         │    Traditional      │       CursedNet           │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Total FLOPs           │    ~40-50 MFLOPs    │     ~26,700 MFLOPs        │
│ Ratio                 │        1×           │        ~530-670×          │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Parameters            │     ~200 (filter    │     ~117,500              │
│                       │      coefficients)  │                           │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Memory (weights)      │     ~1.6 KB         │     ~470 KB (FP32)        │
│                       │                     │     ~235 KB (FP16)        │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Memory (activations)  │     ~2 MB           │     ~200 MB (peak)        │
│ (inference buffer)    │     (streaming)     │     (batch mode)          │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Latency (theoretical) │     ~4 μs/sample    │     ~104 ms/chunk         │
│                       │     (streaming)     │     (batch, 1 sec)        │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Throughput (CPU)      │     >10 MHz         │     ~2.5 MHz              │
│ (single core)         │     (real-time+)    │     (real-time capable)   │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Throughput (GPU)      │       N/A           │     ~50-100 MHz           │
│ (RTX 3080)            │                     │     (highly parallel)     │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Power (typical)       │     ~1-5 W          │     ~50-150 W (GPU)       │
│                       │     (embedded)      │     ~10-30 W (CPU)        │
├───────────────────────┼─────────────────────┼───────────────────────────┤
│ Implementation        │     Fixed-point     │     Floating-point        │
│                       │     FPGA/DSP ready  │     GPU optimized         │
└───────────────────────┴─────────────────────┴───────────────────────────┘
```

### Detailed Breakdown by Stage

#### Phase Detection Comparison

**Traditional (CORDIC algorithm):**
```
Input: I[n], Q[n] (complex sample)
Output: φ[n] = atan2(Q, I)

CORDIC iterations (typically 16-20):
  - 2 shifts
  - 2 additions
  - 1 table lookup
  = ~5 ops × 16 iterations = 80 ops/sample

For 256,000 samples: 20.5 MFLOPs
```

**Neural equivalent (Conv1 layers):**
```
The first conv_block learns implicit phase relationships:

Conv1a: 2 → 32 channels
  FLOPs = 2 × 3 × 2 × 32 × 256,000 = 98.3 MFLOPs

Conv1b: 32 → 32 channels
  FLOPs = 2 × 3 × 32 × 32 × 256,000 = 1,573 MFLOPs

Total for phase-equivalent: ~1,671 MFLOPs (81× more expensive)
```

#### Differentiation Comparison

**Traditional:**
```
Simple: φ[n] - φ[n-1] with phase unwrapping
  - 1 subtraction
  - 1-3 comparisons for unwrap
  = ~4 ops/sample

For 256,000 samples: 1.0 MFLOPs

Or FIR differentiator (5-tap):
  = 10 ops/sample = 2.6 MFLOPs
```

**Neural equivalent (early layers with kernel=3):**
```
Convolution kernel [a, b, c] can learn derivative:
  Ideal derivative kernel ≈ [-0.5, 0, 0.5]

Already counted in Conv1/Conv2 layers
```

#### Filtering Comparison

**Traditional De-emphasis (1st order IIR):**
```
y[n] = α×x[n] + (1-α)×y[n-1]
  - 2 multiplications
  - 1 addition
  = 3 ops/sample

For 256,000 samples: 0.77 MFLOPs
```

**Traditional Anti-alias (64-tap FIR, polyphase):**
```
With 8:1 decimation, polyphase implementation:
  - 8 phases, each with 8 taps
  - Only compute 32,000 outputs
  = 8 × 2 × 32,000 = 512K ops = 0.5 MFLOPs

Naive (no polyphase):
  = 64 × 2 × 256,000 = 32.8 MFLOPs
```

**Neural equivalent (Conv4-Conv7 + MaxPool):**
```
These layers perform learned filtering and decimation:

Conv4-Conv5 @ 128k: ~2,949 MFLOPs
Conv6-Conv7 @ 64k:  ~92 MFLOPs
MaxPool layers:     ~9 MFLOPs

Total: ~3,050 MFLOPs (filtering equivalent)
```

### Memory Access Patterns

#### Traditional Chain

```
Memory Bandwidth Requirements:
┌─────────────────────────────────────────────────────────────┐
│ Stage          │ Reads/sample │ Writes/sample │ Pattern     │
├─────────────────────────────────────────────────────────────┤
│ Phase detect   │ 2 (I, Q)     │ 1 (φ)         │ Sequential  │
│ Differentiate  │ 2 (φ[n-1:n]) │ 1             │ Sequential  │
│ De-emphasis    │ 2            │ 1             │ Sequential  │
│ FIR filter     │ 64 (taps)    │ 1             │ Sequential  │
│ Decimate       │ 1            │ 1/8           │ Strided     │
├─────────────────────────────────────────────────────────────┤
│ Total          │ ~71          │ ~4.1          │             │
│ @ 256 kHz      │ 18.2 MB/s    │ 1.1 MB/s      │             │
└─────────────────────────────────────────────────────────────┘

Advantages:
- Streaming possible (minimal buffering)
- Cache-friendly sequential access
- Low memory footprint (~10 KB working set)
```

#### CursedNet Neural Network

```
Memory Bandwidth Requirements:
┌─────────────────────────────────────────────────────────────┐
│ Layer          │ Input Size    │ Output Size   │ Weights    │
├─────────────────────────────────────────────────────────────┤
│ Conv1          │ 2×256k = 2 MB │ 32×256k= 32MB │ 6.5 KB     │
│ Conv2          │ 32 MB         │ 64 MB         │ 37 KB      │
│ Conv3          │ 64 MB         │ 64 MB         │ 49 KB      │
│ MaxPool1       │ 64 MB         │ 32 MB         │ 0          │
│ Conv4          │ 32 MB         │ 16 MB         │ 18 KB      │
│ Conv5          │ 16 MB         │ 8 MB          │ 4.6 KB     │
│ MaxPool2       │ 8 MB          │ 4 MB          │ 0          │
│ Conv6          │ 4 MB          │ 2 MB          │ 1.2 KB     │
│ Conv7          │ 2 MB          │ 1 MB          │ 0.3 KB     │
│ MaxPool3       │ 1 MB          │ 0.5 MB        │ 0          │
│ Conv8          │ 0.5 MB        │ 0.13 MB       │ 0.02 KB    │
├─────────────────────────────────────────────────────────────┤
│ Peak Activation Memory: ~200 MB (between Conv2 and Conv3)  │
│ Total Weight Memory: ~470 KB                                │
└─────────────────────────────────────────────────────────────┘

Note: With gradient checkpointing or streaming inference,
activation memory can be reduced significantly.
```

### Latency Analysis

#### Traditional Chain (Sample-by-Sample)

```
Per-sample processing time breakdown:
┌─────────────────────────────────────────────────────────────┐
│ Stage              │ Cycles (est.) │ Time @ 1 GHz           │
├─────────────────────────────────────────────────────────────┤
│ Phase detection    │ 50-100        │ 50-100 ns              │
│ Differentiation    │ 5-10          │ 5-10 ns                │
│ De-emphasis        │ 5             │ 5 ns                   │
│ FIR (polyphase)    │ 20            │ 20 ns                  │
│ Decimation         │ 2             │ 2 ns                   │
├─────────────────────────────────────────────────────────────┤
│ Total per sample   │ ~82-137       │ ~82-137 ns             │
│ Throughput         │               │ 7.3-12.2 MHz           │
└─────────────────────────────────────────────────────────────┘

Latency: ~1 sample = 3.9 μs @ 256 kHz
Can process in real-time with margin
```

#### CursedNet (Batch Processing)

```
GPU Processing (RTX 3080 - 8704 CUDA cores @ 1.71 GHz):
┌─────────────────────────────────────────────────────────────┐
│ Metric                    │ Value                           │
├─────────────────────────────────────────────────────────────┤
│ Peak TFLOPS (FP32)        │ 29.8 TFLOPS                     │
│ Memory Bandwidth          │ 760 GB/s                        │
│ CursedNet FLOPs           │ 26.7 GFLOPs                     │
│ Theoretical time          │ 0.9 ms (compute bound)          │
│ Actual time (estimated)   │ 5-15 ms (memory bound)          │
├─────────────────────────────────────────────────────────────┤
│ Throughput                │ 67-200 chunks/sec               │
│                           │ = 17-51 MHz equivalent          │
│ Latency                   │ 5-15 ms + 1 sec buffer          │
│                           │ ≈ 1 second (batch size)         │
└─────────────────────────────────────────────────────────────┘

CPU Processing (i7-10700K @ 3.8 GHz, 8 cores):
┌─────────────────────────────────────────────────────────────┐
│ Peak GFLOPS (AVX2)        │ ~400 GFLOPS (theoretical)       │
│ Realistic (CNN)           │ ~50-100 GFLOPS                  │
│ CursedNet time            │ 270-530 ms per chunk            │
│ Throughput                │ 1.9-3.7 chunks/sec              │
│                           │ = 0.5-0.9 MHz equivalent        │
└─────────────────────────────────────────────────────────────┘

With batch size 1, streaming is NOT real-time on CPU.
Requires chunked/overlapped processing or GPU.
```

### Hardware Implementation Comparison

#### FPGA/ASIC (Traditional)

```
Traditional FM Demod on FPGA (Xilinx Zynq-7020):
┌─────────────────────────────────────────────────────────────┐
│ Resource          │ Usage        │ Notes                    │
├─────────────────────────────────────────────────────────────┤
│ LUTs              │ ~2,000       │ CORDIC + filters         │
│ DSP Slices        │ 8-16         │ FIR filter               │
│ BRAM              │ 2-4          │ Coefficient storage      │
│ Clock             │ 100-250 MHz  │ Easily achievable        │
│ Throughput        │ 100-250 MSPS │ 1 sample/clock possible  │
│ Latency           │ ~50-100 ns   │ Pipeline depth           │
│ Power             │ ~0.5-2 W     │ Very efficient           │
└─────────────────────────────────────────────────────────────┘
```

#### GPU/NPU (Neural)

```
CursedNet on Various Platforms:
┌─────────────────────────────────────────────────────────────┐
│ Platform          │ Throughput    │ Latency   │ Power      │
├─────────────────────────────────────────────────────────────┤
│ RTX 3080          │ 50-100 MHz    │ ~15 ms    │ 150-320 W  │
│ Jetson Xavier NX  │ 5-10 MHz      │ ~100 ms   │ 10-15 W    │
│ Intel NCS2        │ 0.5-1 MHz     │ ~1 sec    │ ~1 W       │
│ Coral Edge TPU    │ 1-2 MHz       │ ~500 ms   │ ~2 W       │
│ Apple M1 Neural   │ 10-20 MHz     │ ~50 ms    │ ~10 W      │
└─────────────────────────────────────────────────────────────┘
```

### Efficiency Metrics

```
┌─────────────────────────────────────────────────────────────┐
│              Efficiency Comparison                          │
├───────────────────────┬─────────────────┬───────────────────┤
│ Metric                │ Traditional     │ CursedNet         │
├───────────────────────┼─────────────────┼───────────────────┤
│ FLOPs per output      │ ~1,250          │ ~835,000          │
│ sample                │                 │ (668× more)       │
├───────────────────────┼─────────────────┼───────────────────┤
│ Bytes per output      │ ~280 B          │ ~6,250 B          │
│ (memory traffic)      │                 │ (22× more)        │
├───────────────────────┼─────────────────┼───────────────────┤
│ Energy per output     │ ~0.01 nJ        │ ~1-5 nJ           │
│ (estimated)           │                 │ (100-500× more)   │
├───────────────────────┼─────────────────┼───────────────────┤
│ Parameters per        │ 0.006           │ 3.7               │
│ output sample         │                 │ (617× more)       │
└───────────────────────┴─────────────────┴───────────────────┘
```

### When to Use Each Approach

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| **Embedded/IoT** | Traditional | Lower power, smaller footprint |
| **Real-time SDR** | Traditional | Lower latency, streaming |
| **Research/Prototyping** | Neural | Flexibility, easy modification |
| **Multi-mode demod** | Neural | Single model can learn multiple schemes |
| **Noisy environments** | Neural | Can learn noise rejection |
| **FPGA deployment** | Traditional | Well-established IP cores |
| **GPU server** | Neural | Parallel batch processing |
| **Unknown modulation** | Neural | Blind demodulation capability |

### Optimization Opportunities for CursedNet

To improve computational efficiency:

1. **Quantization**: INT8 reduces compute by 4×, memory by 4×
   - Expected: ~6.7 GFLOPs INT8 equivalent

2. **Pruning**: Remove 50-80% of weights
   - Expected: 2-5× speedup with sparse ops

3. **Knowledge Distillation**: Train smaller student network
   - Target: ~1-5 GFLOPs with minimal quality loss

4. **Architecture Search**: Find optimal channel widths
   - Current architecture may be over-parameterized

5. **Streaming Convolutions**: Process overlapping chunks
   - Reduce latency from 1 sec to ~10-50 ms

```python
# Example: Streaming inference with overlap
class StreamingCursedNet:
    def __init__(self, model, chunk_size=32000, overlap=1000):
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.buffer = None

    def process_stream(self, new_samples):
        # Append to buffer
        if self.buffer is not None:
            samples = np.concatenate([self.buffer, new_samples])
        else:
            samples = new_samples

        # Process complete chunks
        outputs = []
        while len(samples) >= self.chunk_size * 8:  # RF samples
            chunk = samples[:self.chunk_size * 8]
            output = self.model(chunk)
            outputs.append(output[:-self.overlap//8])  # Remove overlap
            samples = samples[self.chunk_size * 8 - self.overlap * 8:]

        self.buffer = samples
        return np.concatenate(outputs) if outputs else np.array([])
```

### Conclusion

The neural approach (CursedNet) is approximately **500-700× more computationally expensive** than traditional FM demodulation for the same task. However, it offers:

1. **Flexibility**: Can adapt to different conditions through training
2. **Blind operation**: No explicit parameter tuning required
3. **Potential for multi-task**: Could learn multiple modulation schemes
4. **GPU acceleration**: Highly parallelizable on modern hardware

The traditional approach remains superior for:
1. **Efficiency**: Orders of magnitude less computation
2. **Latency**: Sample-by-sample streaming possible
3. **Embedded deployment**: FPGA/DSP friendly
4. **Power consumption**: Much lower energy per sample

**Recommendation**: Use traditional demodulation for production systems where the modulation scheme is known. Use neural approaches for research, blind demodulation scenarios, or when learning complex channel characteristics is beneficial.

---

## References

1. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
2. **FM Demodulation**: Carlson, "Communication Systems" (4th ed.)
3. **Neural Audio Processing**: Engel et al., "DDSP: Differentiable Digital Signal Processing" (2020)
4. **CORDIC Algorithm**: Volder, "The CORDIC Trigonometric Computing Technique" (1959)
5. **Efficient CNN**: Howard et al., "MobileNets: Efficient Convolutional Neural Networks" (2017)
