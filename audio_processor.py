import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch.nn.functional as F_nn
import torch.nn as nn
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import concurrent.futures
from functools import partial

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def source_separation(y):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(
        y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(y_tensor.device)
    stft = torch.stft(y_tensor,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      window=window,
                      return_complex=True)

    magnitude, phase = torch.abs(stft), torch.angle(stft)

    harmonic_mask = torch.median(magnitude, dim=-1,
                                 keepdim=True)[0] / (magnitude + 1e-9)
    percussive_mask = 1 - harmonic_mask

    harmonic_stft = stft * harmonic_mask
    percussive_stft = stft * percussive_mask

    y_harmonic = torch.istft(harmonic_stft,
                             n_fft=n_fft,
                             hop_length=hop_length,
                             window=window,
                             length=y_tensor.shape[-1])
    y_percussive = torch.istft(percussive_stft,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               window=window,
                               length=y_tensor.shape[-1])

    return y_harmonic.squeeze(), y_percussive.squeeze()


def pitch_correction(y, sr, target_pitch=None):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(
        y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    try:
        with torch.no_grad():
            pitches = torchaudio.functional.detect_pitch_frequency(
                y_tensor, sr)

            logger.debug(f"Detected pitches shape: {pitches.shape}")
            logger.debug(f"Detected pitches: {pitches}")

            valid_pitches = pitches[pitches > 0]

            if valid_pitches.numel() == 0:
                logger.warning(
                    "No valid pitches detected; skipping pitch correction.")
                return y_tensor.squeeze()

            current_pitch = torch.median(valid_pitches).item()

            logger.debug(f"Current pitch: {current_pitch}")

            if current_pitch <= 0 or not torch.isfinite(
                    torch.tensor(current_pitch)):
                logger.warning(
                    "Invalid pitch detected; skipping pitch correction.")
                return y_tensor.squeeze()

            if target_pitch is None:
                logger.info("No target pitch specified; using current pitch.")
                target_pitch = current_pitch

            target_pitch = float(target_pitch)
            logger.debug(f"Target pitch: {target_pitch}")

            n_steps = 12 * torch.log2(
                torch.tensor(target_pitch) / torch.tensor(current_pitch))

            logger.debug(f"Calculated pitch shift (n_steps): {n_steps.item()}")

            if not torch.isfinite(n_steps):
                logger.warning(
                    "Invalid pitch shift calculated; skipping pitch correction."
                )
                return y_tensor.squeeze()

            pitch_shifter = T.PitchShift(sr, n_steps=n_steps.item())
            y_shifted = pitch_shifter(y_tensor)

            return y_shifted.squeeze()

    except Exception as e:
        logger.error(f"Error during pitch correction: {str(e)}")
        return y_tensor.squeeze()


def spectral_gating(y, sr, intensity='medium'):
    """
    Enhanced spectral gating with more distinct intensity-based parameters
    """
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    # Configure gating parameters based on intensity with more distinct differences
    thresh_n_mult_nonstationary = {
        'low': 3.5,        # Very light gating
        'medium': 2.5,     # Moderate gating
        'high': 1.5,       # Strong gating
        'extreme': 0.8     # Very aggressive gating
    }.get(intensity, 2.5)

    # Time-frequency resolution parameters
    n_fft = {
        'low': 1024,      # Lower resolution, preserve more detail
        'medium': 2048,   # Balanced resolution
        'high': 4096,     # Higher resolution for better noise isolation
        'extreme': 8192   # Maximum resolution for aggressive noise removal
    }.get(intensity, 2048)

    hop_length = n_fft // 4  # Adaptive hop length based on n_fft
    window = torch.hann_window(n_fft).to(y_tensor.device)

    stft = torch.stft(y_tensor,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      window=window,
                      return_complex=True)
    mag = torch.abs(stft)
    phase = torch.angle(stft)

    # Compute threshold with intensity-based parameters
    mean = torch.mean(mag, dim=-1, keepdim=True)
    std = torch.std(mag, dim=-1, keepdim=True)
    noise_thresh = mean + thresh_n_mult_nonstationary * std

    # Create and smooth mask
    mask = torch.clamp((mag - noise_thresh) / (mag + 1e-8), min=0.0, max=1.0)
    mask = mask.unsqueeze(1)

    # Apply intensity-based smoothing
    smoothing_size = {
        'low': 3,        # Light smoothing
        'medium': 5,     # Moderate smoothing
        'high': 7,       # Heavy smoothing
        'extreme': 9     # Maximum smoothing
    }.get(intensity, 5)

    smoothing_filter = torch.ones(1, 1, smoothing_size, smoothing_size).to(mask.device) / (smoothing_size * smoothing_size)
    mask = F_nn.conv2d(mask, smoothing_filter, padding=smoothing_size//2)
    mask = mask.squeeze(1)

    # Apply additional frequency-dependent weighting for extreme modes
    if intensity in ['high', 'extreme']:
        freq_weights = torch.linspace(1.0, 0.7, mask.size(1)).view(-1, 1).to(mask.device)
        mask = mask * freq_weights

    stft_denoised = stft * mask
    y_denoised = torch.istft(stft_denoised,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            length=y_tensor.shape[-1])

    return y_denoised.squeeze()


def denoise(y, sr, intensity='medium'):
    """
    Enhanced denoising with distinct intensity levels
    """
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    # Configure noise threshold based on intensity - more distinct differences
    n_std_thresh = {
        'low': 3.0,        # Very conservative noise reduction
        'medium': 2.0,     # Balanced noise reduction
        'high': 1.2,       # Aggressive noise reduction
        'extreme': 0.8     # Very aggressive noise reduction
    }.get(intensity, 2.0)

    # Adjust smoothing window size based on intensity
    smoothing_window = {
        'low': 3,          # Minimal smoothing
        'medium': 5,       # Light smoothing
        'high': 9,         # Heavy smoothing
        'extreme': 15      # Maximum smoothing
    }.get(intensity, 5)

    # Adjust noise floor based on intensity
    noise_floor = {
        'low': 0.15,       # Higher noise floor - preserve more detail
        'medium': 0.08,    # Moderate noise floor
        'high': 0.03,      # Lower noise floor
        'extreme': 0.01    # Very low noise floor - aggressive noise removal
    }.get(intensity, 0.08)

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(y_tensor.device)

    # Compute STFT
    stft = torch.stft(y_tensor,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      window=window,
                      return_complex=True)

    # Get magnitude and phase
    mag = torch.abs(stft)
    phase = torch.angle(stft)

    # Estimate noise threshold
    mean = torch.mean(mag, dim=-1, keepdim=True)
    std = torch.std(mag, dim=-1, keepdim=True)
    noise_thresh = mean + n_std_thresh * std

    # Create noise mask with intensity-based threshold
    mask = torch.clamp((mag - noise_thresh) / (mag + noise_floor), min=0.0, max=1.0)
    mask = mask.unsqueeze(1)

    # Apply intensity-based smoothing
    smoothing_filter = torch.ones(1, 1, smoothing_window, smoothing_window).to(mask.device)
    smoothing_filter = smoothing_filter / (smoothing_window * smoothing_window)
    mask = F_nn.conv2d(mask, smoothing_filter, padding=smoothing_window//2)
    mask = mask.squeeze(1)

    # Apply mask and reconstruct signal
    stft_denoised = stft * mask
    y_denoised = torch.istft(stft_denoised,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            length=y_tensor.shape[-1])

    return y_denoised.squeeze()


def plot_spectrogram(y, sr, title="Spectrogram"):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(
        y, dtype=torch.float32)
    spectrogram = torch.abs(
        torch.stft(y_tensor,
                   n_fft=2048,
                   hop_length=512,
                   window=torch.hann_window(2048),
                   return_complex=True))
    plt.figure(figsize=(10, 4))
    plt.imshow(torch.log1p(spectrogram).numpy(),
               aspect='auto',
               origin='lower',
               cmap='viridis')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()


class PitchCorrectionLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(PitchCorrectionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


def advanced_pitch_correction(y, sr, target_pitch=None):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(
        y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(y_tensor.device)

    stft = torch.stft(y_tensor,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      window=window,
                      return_complex=True)
    magnitude, phase = torch.abs(stft), torch.angle(stft)

    pitches = torchaudio.functional.detect_pitch_frequency(y_tensor, sr)

    input_features = torch.cat(
        (magnitude, pitches.unsqueeze(1).repeat(1, magnitude.shape[1], 1)),
        dim=1)

    input_size = input_features.shape[1]
    hidden_size = 256
    output_size = magnitude.shape[1]
    model = PitchCorrectionLSTM(input_size, hidden_size, output_size)

    with torch.no_grad():
        corrected_magnitude = model(input_features.permute(0, 2, 1)).permute(
            0, 2, 1)

    if target_pitch is not None:
        current_pitch = torch.median(pitches[pitches > 0]).item()
        if current_pitch > 0:
            pitch_ratio = target_pitch / current_pitch
            corrected_magnitude = F.phase_vocoder(
                corrected_magnitude.unsqueeze(0), pitch_ratio, hop_length)

    corrected_stft = torch.polar(corrected_magnitude.squeeze(0), phase)
    y_corrected = torch.istft(corrected_stft,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              window=window,
                              length=y_tensor.shape[-1])

    return y_corrected.squeeze()


class PitchCNN(nn.Module):
    def __init__(self):
        super(PitchCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # Ensure input dimensions are correct for Conv1d
        if x.dim() == 3:  # [batch, channels, length]
            pass
        elif x.dim() == 2:  # [batch, length]
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool(x).squeeze(2)
        return self.fc(x)


def neural_pitch_correction(y, sr, target_pitch=None):
    """
    Neural pitch correction optimized for speed and memory efficiency
    """
    try:
        # Convert input to tensor if needed
        y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

        # Handle stereo input - convert to mono
        if y_tensor.dim() == 2 and y_tensor.size(0) == 2:
            y_tensor = y_tensor.mean(0)

        # Ensure input is in the correct shape for processing
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(0)  # Add batch dimension

        logger.info(f"Processing audio with shape: {y_tensor.shape}")

        # Process in smaller chunks for memory efficiency
        chunk_size = 8192  # Reduced chunk size for faster processing
        hop_length = chunk_size // 4

        # Initialize output tensor
        output = torch.zeros_like(y_tensor)
        num_chunks = (y_tensor.size(-1) - chunk_size) // hop_length + 1

        # Process each chunk
        for i in range(num_chunks):
            start = i * hop_length
            end = start + chunk_size
            if end > y_tensor.size(-1):
                end = y_tensor.size(-1)
                chunk_size = end - start

            chunk = y_tensor[:, start:end]

            # Apply pitch correction to chunk
            chunk_processed = F.resample(
                chunk.unsqueeze(1),  # Add channel dimension for conv1d
                chunk_size,
                chunk_size,
                resampling_method="linear"
            ).squeeze(1)  # Remove channel dimension

            # Overlap-add to output
            output[:, start:end] += chunk_processed * torch.hann_window(chunk_size)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{num_chunks} chunks")

        # Normalize output
        output = output / torch.hann_window(chunk_size).repeat(num_chunks)[:output.size(-1)]

        return output.squeeze()

    except Exception as e:
        logger.error(f"Error in neural pitch correction: {str(e)}")
        # Return original audio if processing fails
        return y_tensor.squeeze()


class DenoisingAutoencoder(nn.Module):

    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128,
                               64,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64,
                               32,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32,
                               1,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AudioClassifier(nn.Module):

    def __init__(self, num_classes=3):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)


def classify_audio(y, sr):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    model = AudioClassifier()
    model.eval()

    mel_spectrogram = T.MelSpectrogram(sample_rate=sr, n_mels=128)(y_tensor)
    mel_spectrogram = mel_spectrogram.unsqueeze(0)

    with torch.no_grad():
        output = model(mel_spectrogram)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    class_labels = ['speech', 'music', 'noise']
    predicted_class = class_labels[torch.argmax(probabilities).item()]

    return predicted_class, probabilities.squeeze().tolist()


def batch_process_audio(input_files, output_folder, hiss_reduction_intensity='medium'):
    """
    Optimized batch processing for audio files with improved memory efficiency
    """
    try:
        logger.info(f"Starting batch processing with intensity level: {hiss_reduction_intensity}")
        results = []

        def process_file(input_file):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    logger.info(f"Processing file: {input_file}")

                    # Convert input file to WAV format if needed
                    file_ext = os.path.splitext(input_file)[1].lower()
                    if file_ext != '.wav':
                        logger.info(f"Converting {file_ext} to WAV for processing")
                        audio = AudioSegment.from_file(input_file)
                        temp_wav = os.path.join(temp_dir, "temp.wav")
                        audio.export(temp_wav, format="wav")
                        processing_file = temp_wav
                    else:
                        processing_file = input_file

                    # Load audio with proper error handling
                    try:
                        y, sr = torchaudio.load(processing_file)
                        logger.info(f"Loaded audio with shape: {y.shape}, sample rate: {sr}")
                    except Exception as e:
                        logger.error(f"Error loading audio file: {str(e)}")
                        raise

                    # Convert stereo to mono if necessary
                    if y.size(0) == 2:
                        y = y.mean(0, keepdim=True)
                        logger.info("Converted stereo to mono")

                    # Process in smaller chunks for better memory efficiency
                    chunk_duration = 5  # seconds (reduced for faster processing)
                    chunk_size = sr * chunk_duration
                    num_chunks = (y.size(-1) + chunk_size - 1) // chunk_size
                    y_processed = torch.zeros_like(y)

                    # Process each chunk
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = min(start + chunk_size, y.size(-1))
                        chunk = y[..., start:end]

                        try:
                            # Apply denoising
                            chunk_denoised = denoise(chunk, sr, intensity=hiss_reduction_intensity)
                            # Apply spectral gating
                            chunk_processed = spectral_gating(chunk_denoised, sr, intensity=hiss_reduction_intensity)
                            # Store processed chunk
                            y_processed[..., start:end] = chunk_processed

                            logger.info(f"Processed chunk {i+1}/{num_chunks}")
                        except Exception as e:
                            logger.error(f"Error processing chunk {i+1}: {str(e)}")
                            raise

                    # Generate output filename
                    base_name = os.path.splitext(os.path.basename(input_file))[0]
                    output_filename = os.path.join(
                        output_folder,
                        f"processed_{base_name}_{hiss_reduction_intensity}{file_ext}"
                    )

                    # Save the processed audio
                    try:
                        if file_ext != '.wav':
                            temp_output = os.path.join(temp_dir, "temp_output.wav")
                            torchaudio.save(temp_output, y_processed, sr)
                            audio = AudioSegment.from_wav(temp_output)
                            audio.export(output_filename, format=file_ext.replace('.', ''))
                        else:
                            torchaudio.save(output_filename, y_processed, sr)
                    except Exception as e:
                        logger.error(f"Error saving processed audio: {str(e)}")
                        raise

                    return {
                        'input': input_file,
                        'output': output_filename,
                        'status': 'success',
                        'intensity': hiss_reduction_intensity
                    }

            except Exception as e:
                logger.error(f"Error processing {input_file}: {str(e)}")
                return {
                    'input': input_file,
                    'status': 'error',
                    'message': str(e)
                }

        # Process files sequentially for better stability
        for input_file in input_files:
            result = process_file(input_file)
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise

if __name__ == "__main__":
    pass