import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch.nn.functional as F_nn
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from torch import nn

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def source_separation(y):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(y_tensor.device)
    stft = torch.stft(y_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

    magnitude, phase = torch.abs(stft), torch.angle(stft)

    harmonic_mask = torch.median(magnitude, dim=-1, keepdim=True)[0] / (magnitude + 1e-9)
    percussive_mask = 1 - harmonic_mask

    harmonic_stft = stft * harmonic_mask
    percussive_stft = stft * percussive_mask

    y_harmonic = torch.istft(harmonic_stft, n_fft=n_fft, hop_length=hop_length, window=window, length=y_tensor.shape[-1])
    y_percussive = torch.istft(percussive_stft, n_fft=n_fft, hop_length=hop_length, window=window, length=y_tensor.shape[-1])

    return y_harmonic.squeeze(), y_percussive.squeeze()

def pitch_correction(y, sr, target_pitch=None):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    try:
        with torch.no_grad():
            pitches = torchaudio.functional.detect_pitch_frequency(y_tensor, sr)
            
            logger.debug(f"Detected pitches shape: {pitches.shape}")
            logger.debug(f"Detected pitches: {pitches}")

            valid_pitches = pitches[pitches > 0]

            if valid_pitches.numel() == 0:
                logger.warning("No valid pitches detected; skipping pitch correction.")
                return y_tensor.squeeze()

            current_pitch = torch.median(valid_pitches).item()

            logger.debug(f"Current pitch: {current_pitch}")

            if current_pitch <= 0 or not torch.isfinite(torch.tensor(current_pitch)):
                logger.warning("Invalid pitch detected; skipping pitch correction.")
                return y_tensor.squeeze()

            if target_pitch is None:
                logger.info("No target pitch specified; using current pitch.")
                target_pitch = current_pitch

            target_pitch = float(target_pitch)
            logger.debug(f"Target pitch: {target_pitch}")

            n_steps = 12 * torch.log2(torch.tensor(target_pitch) / torch.tensor(current_pitch))

            logger.debug(f"Calculated pitch shift (n_steps): {n_steps.item()}")

            if not torch.isfinite(n_steps):
                logger.warning("Invalid pitch shift calculated; skipping pitch correction.")
                return y_tensor.squeeze()

            pitch_shifter = T.PitchShift(sr, n_steps=n_steps.item())
            y_shifted = pitch_shifter(y_tensor)

            return y_shifted.squeeze()

    except Exception as e:
        logger.error(f"Error during pitch correction: {str(e)}")
        return y_tensor.squeeze()

def spectral_gating(y, sr, n_std_thresh=1.5):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(y_tensor.device)

    stft = torch.stft(y_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mag = torch.abs(stft)
    phase = torch.angle(stft)

    mean = torch.mean(mag, dim=-1, keepdim=True)
    std = torch.std(mag, dim=-1, keepdim=True)
    noise_thresh = mean + n_std_thresh * std

    mask = torch.clamp((mag - noise_thresh) / (mag + 1e-8), min=0.0, max=1.0)
    mask = mask.unsqueeze(1)

    smoothing_filter = torch.ones(1, 1, 5, 5).to(mask.device) / 25
    mask = F_nn.conv2d(mask, smoothing_filter, padding=2)
    mask = mask.squeeze(1)

    stft_denoised = stft * mask
    y_denoised = torch.istft(stft_denoised, n_fft=n_fft, hop_length=hop_length, window=window, length=y_tensor.shape[-1])

    return y_denoised.squeeze()

def denoise(y, sr, intensity='medium'):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    n_std_thresh = {
        'low': 2.5,
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

class PitchCorrectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(PitchCorrectionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

def advanced_pitch_correction(y, sr, target_pitch=None):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(y_tensor.device)

    stft = torch.stft(y_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    magnitude, phase = torch.abs(stft), torch.angle(stft)

    pitches = torchaudio.functional.detect_pitch_frequency(y_tensor, sr)
    
    input_features = torch.cat((magnitude, pitches.unsqueeze(1).repeat(1, magnitude.shape[1], 1)), dim=1)
    
    input_size = input_features.shape[1]
    hidden_size = 256
    output_size = magnitude.shape[1]
    model = PitchCorrectionLSTM(input_size, hidden_size, output_size)
    
    with torch.no_grad():
        corrected_magnitude = model(input_features.permute(0, 2, 1)).permute(0, 2, 1)
    
    if target_pitch is not None:
        current_pitch = torch.median(pitches[pitches > 0]).item()
        if current_pitch > 0:
            pitch_ratio = target_pitch / current_pitch
            corrected_magnitude = F.phase_vocoder(corrected_magnitude.unsqueeze(0), pitch_ratio, hop_length)
    
    corrected_stft = torch.polar(corrected_magnitude.squeeze(0), phase)
    y_corrected = torch.istft(corrected_stft, n_fft=n_fft, hop_length=hop_length, window=window, length=y_tensor.shape[-1])

    return y_corrected.squeeze()

def batch_process_audio(input_files, output_folder, hiss_reduction_intensity='medium'):
    results = []
    for input_file in input_files:
        try:
            y, sr = torchaudio.load(input_file)
            
            y_denoised = denoise(y, sr, intensity=hiss_reduction_intensity)
            y_pitch_corrected = pitch_correction(y_denoised, sr)
            y_harmonic, y_percussive = source_separation(y_pitch_corrected)
            
            y_processed = y_harmonic + y_percussive
            
            output_filename = os.path.join(output_folder, f"processed_{os.path.basename(input_file)}")
            
            torchaudio.save(output_filename, y_processed, sr)
            
            results.append({
                'input': input_file,
                'output': output_filename,
                'status': 'success'
            })
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            results.append({
                'input': input_file,
                'status': 'error',
                'message': str(e)
            })
    
    return results

if __name__ == "__main__":
    pass