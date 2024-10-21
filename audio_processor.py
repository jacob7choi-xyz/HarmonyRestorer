import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
import logging
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def source_separation(y):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.from_numpy(y).float()
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor
    
    stft = torch.stft(y_tensor, n_fft=2048, hop_length=512, window=torch.hann_window(2048).to(y_tensor.device), return_complex=True)
    
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    harmonic = torch.median(magnitude, dim=-1, keepdim=True).values
    percussive = torch.median(magnitude, dim=-2, keepdim=True).values
    
    harmonic_mask = (harmonic / (harmonic + percussive + 1e-8)).pow(2)
    percussive_mask = (percussive / (harmonic + percussive + 1e-8)).pow(2)
    
    harmonic_complex = torch.polar(magnitude * harmonic_mask, phase)
    percussive_complex = torch.polar(magnitude * percussive_mask, phase)
    
    y_harmonic = torch.istft(harmonic_complex, n_fft=2048, hop_length=512, window=torch.hann_window(2048).to(y_tensor.device), length=y_tensor.shape[-1])
    y_percussive = torch.istft(percussive_complex, n_fft=2048, hop_length=512, window=torch.hann_window(2048).to(y_tensor.device), length=y_tensor.shape[-1])
    
    return y_harmonic.squeeze(), y_percussive.squeeze()

def pitch_correction(y, sr, target_pitch=None):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    try:
        with torch.no_grad():
            pitches = torchaudio.functional.detect_pitch_frequency(y_tensor, sr)
            
            logger.debug(f"Detected pitches: {pitches}")

            valid_pitches = pitches[pitches > 0]

            if valid_pitches.numel() == 0:
                logger.warning("No valid pitches detected; skipping pitch correction.")
                return y_tensor.squeeze()

            current_pitch = torch.median(valid_pitches).item()

            if current_pitch <= 0 or not torch.isfinite(torch.tensor(current_pitch)):
                logger.warning("Invalid pitch detected; skipping pitch correction.")
                return y_tensor.squeeze()

            if target_pitch is None:
                target_pitch = current_pitch

            target_pitch = float(target_pitch)

            n_steps = 12 * torch.log2(torch.tensor(target_pitch) / torch.tensor(current_pitch))

            if not torch.isfinite(n_steps):
                logger.warning("Invalid pitch shift calculated; skipping pitch correction.")
                return y_tensor.squeeze()

            logger.debug(f"Calculated pitch shift (n_steps): {n_steps.item()}")

            # Apply pitch shift using torchaudio's pitch_shift function
            y_shifted = torchaudio.functional.pitch_shift(y_tensor, sr, n_steps.item())

            return y_shifted.squeeze()

    except Exception as e:
        logger.error(f"Error during pitch correction: {str(e)}")
        return y_tensor.squeeze()

def spectral_gating(y, sr, n_std_thresh=1.5, noise_reduction_factor=2, noise_estimation_sec=0.5):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(y_tensor.device)

    stft = torch.stft(y_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mag = torch.abs(stft)
    phase = torch.angle(stft)

    num_noise_samples = int(noise_estimation_sec * sr / hop_length)
    noise_mag = mag[..., :num_noise_samples]
    
    mean_noise_mag = torch.mean(noise_mag, dim=-1, keepdim=True)
    std_noise_mag = torch.std(noise_mag, dim=-1, keepdim=True)
    
    noise_thresh = torch.max(mean_noise_mag + n_std_thresh * std_noise_mag, mag.mean(dim=-1, keepdim=True) * 0.1)

    mask = torch.clamp((mag - noise_thresh) / (noise_thresh * noise_reduction_factor), min=0.0, max=1.0)
    mask = mask ** 2

    smoothing_filter = torch.ones(1, 1, 3, 3).to(mask.device) / 9
    mask = torch.nn.functional.conv2d(mask.unsqueeze(1), smoothing_filter, padding=1).squeeze(1)

    noise_floor = 0.01
    final_mask = mask * (1 - noise_floor) + noise_floor

    stft_denoised = stft * final_mask
    y_denoised = torch.istft(stft_denoised, n_fft=n_fft, hop_length=hop_length, window=window, length=y_tensor.shape[-1])

    return y_denoised.squeeze()

def denoise(y, sr, intensity='medium'):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor
    
    n_std_thresh = {
        'low': 2.0,
        'medium': 1.5,
        'high': 1.0,
        'extreme': 0.5
    }.get(intensity, 1.5)
    
    return spectral_gating(y_tensor, sr, n_std_thresh)

def plot_spectrogram(y, sr, title="Spectrogram"):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    spectrogram = torch.abs(torch.stft(y_tensor, n_fft=2048, hop_length=512, window=torch.hann_window(2048), return_complex=True))
    plt.figure(figsize=(10, 4))
    plt.imshow(torch.log1p(spectrogram).numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

if __name__ == "__main__":
    pass
