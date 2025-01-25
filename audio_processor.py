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
    Enhanced spectral gating with intensity-based parameters
    """
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(
        y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    # Configure gating parameters based on intensity
    thresh_n_mult_nonstationary = {
        'low': 2.5,
        'medium': 2.0,
        'high': 1.5,
        'extreme': 1.0
    }.get(intensity, 2.0)

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(y_tensor.device)

    stft = torch.stft(y_tensor,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      window=window,
                      return_complex=True)
    mag = torch.abs(stft)
    phase = torch.angle(stft)

    mean = torch.mean(mag, dim=-1, keepdim=True)
    std = torch.std(mag, dim=-1, keepdim=True)
    noise_thresh = mean + thresh_n_mult_nonstationary * std

    mask = torch.clamp((mag - noise_thresh) / (mag + 1e-8), min=0.0, max=1.0)
    mask = mask.unsqueeze(1)

    smoothing_filter = torch.ones(1, 1, 5, 5).to(mask.device) / 25
    mask = F_nn.conv2d(mask, smoothing_filter, padding=2)
    mask = mask.squeeze(1)

    stft_denoised = stft * mask
    y_denoised = torch.istft(stft_denoised,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            length=y_tensor.shape[-1])

    return y_denoised.squeeze()


def denoise(y, sr, intensity='medium'):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(
        y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    # Configure noise threshold and smoothing based on intensity level
    n_std_thresh = {
        'low': 2.5,        # More conservative noise reduction
        'medium': 1.5,     # Balanced noise reduction
        'high': 1.0,       # Aggressive noise reduction
        'extreme': 0.5     # Very aggressive noise reduction
    }.get(intensity, 1.5)

    # Adjust smoothing window size based on intensity
    smoothing_window = {
        'low': 3,          # Less smoothing
        'medium': 5,       # Moderate smoothing
        'high': 7,         # More smoothing
        'extreme': 9       # Maximum smoothing
    }.get(intensity, 5)

    # Adjust noise floor based on intensity
    noise_floor = {
        'low': 0.1,        # Higher noise floor
        'medium': 0.05,    # Moderate noise floor
        'high': 0.02,      # Lower noise floor
        'extreme': 0.01    # Very low noise floor
    }.get(intensity, 0.05)

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
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool(x).squeeze(2)
        return self.fc(x)


def neural_pitch_correction(y, sr, target_pitch=None):
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
    y_tensor = y_tensor.unsqueeze(0) if y_tensor.dim() == 1 else y_tensor

    # Ensure correct dimensions for the CNN
    if y_tensor.dim() == 3:  # If [batch, channels, samples]
        y_tensor = y_tensor.squeeze(1)
    elif y_tensor.dim() == 2 and y_tensor.size(0) == 2:  # If stereo
        y_tensor = y_tensor.mean(0, keepdim=True)  # Convert to mono

    model = PitchCNN()
    model.eval()

    with torch.no_grad():
        frame_length = 2048
        hop_length = 512
        frames = y_tensor.unfold(-1, frame_length, hop_length)

        # Add channel dimension for CNN
        frames = frames.unsqueeze(1)

        pitch_estimates = model(frames).squeeze()

        if target_pitch is None:
            target_pitch = pitch_estimates.mean().item()

        ratios = target_pitch / (pitch_estimates + 1e-8)  # Avoid division by zero
        ratios = torch.clamp(ratios, 0.5, 2.0)  # Limit pitch shifting

        output = torch.zeros_like(y_tensor)
        window = torch.hann_window(frame_length)

        for i, ratio in enumerate(ratios):
            frame = frames[:, 0, i] * window  # Apply window function
            shifted_frame = F.resample(
                frame.unsqueeze(1),
                int(frame_length * ratio),
                frame_length
            ).squeeze(1)

            if shifted_frame.size(-1) >= hop_length:
                output[..., i * hop_length:(i + 1) * hop_length] += shifted_frame[..., :hop_length]

    return output.squeeze()


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
    Optimized batch processing for audio files
    """
    import os
    from pydub import AudioSegment
    import tempfile

    logger.info(f"Starting batch processing with intensity level: {hiss_reduction_intensity}")
    results = []

    for input_file in input_files:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Processing file: {input_file} with intensity: {hiss_reduction_intensity}")

                # Convert input file to WAV format if needed
                file_ext = os.path.splitext(input_file)[1].lower()
                if file_ext == '.mp3':
                    logger.info("Converting MP3 to WAV for processing")
                    audio = AudioSegment.from_mp3(input_file)
                    temp_wav = os.path.join(temp_dir, "temp.wav")
                    audio.export(temp_wav, format="wav")
                    processing_file = temp_wav
                else:
                    processing_file = input_file

                # Load and process audio
                y, sr = torchaudio.load(processing_file)
                logger.info(f"Loaded audio with sample rate: {sr}")

                # Optimize for memory usage by processing in smaller chunks
                chunk_size = sr * 30  # Process 30 seconds at a time
                num_chunks = (y.size(-1) + chunk_size - 1) // chunk_size
                y_processed = torch.zeros_like(y)

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, y.size(-1))
                    chunk = y[..., start:end]

                    # Apply processing chain
                    chunk = denoise(chunk, sr, intensity=hiss_reduction_intensity)
                    chunk = spectral_gating(chunk, sr, intensity=hiss_reduction_intensity)
                    chunk = neural_pitch_correction(chunk, sr)
                    y_harmonic, y_percussive = source_separation(chunk)
                    chunk = y_harmonic + 0.5 * y_percussive  # Reduce percussive component

                    # Store processed chunk
                    y_processed[..., start:end] = chunk

                # Generate output filename
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_filename = os.path.join(
                    output_folder, 
                    f"processed_{base_name}_{hiss_reduction_intensity}{file_ext}"
                )

                # Save the processed audio
                if file_ext == '.mp3':
                    temp_output = os.path.join(temp_dir, "temp_output.wav")
                    torchaudio.save(temp_output, y_processed, sr)
                    audio = AudioSegment.from_wav(temp_output)
                    audio.export(output_filename, format="mp3")
                else:
                    torchaudio.save(output_filename, y_processed, sr)

                results.append({
                    'input': input_file,
                    'output': output_filename,
                    'status': 'success',
                    'intensity': hiss_reduction_intensity
                })

                logger.info(f"Successfully processed {input_file} with intensity {hiss_reduction_intensity}")

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