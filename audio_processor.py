import os
import torch
import torchaudio
import torchaudio.functional as F
import logging
import tempfile
from pydub import AudioSegment

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def spectral_gating(y, sr, intensity='medium'):
    """
    Conservative spectral gating focused on analog recording noise while preserving audio quality
    """
    try:
        # Convert input to tensor if needed
        y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(0)

        # Standard STFT parameters
        n_fft = 2048
        hop_length = n_fft // 4
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

        # Estimate noise floor across all frequencies
        # Use the lowest 5% of magnitudes as an estimate of the noise floor
        sorted_mags, _ = torch.sort(mag, dim=-1)
        noise_floor = sorted_mags[..., :int(sorted_mags.size(-1) * 0.05)].mean(dim=-1, keepdim=True)

        # Calculate threshold based on local statistics and noise floor
        mean = torch.mean(mag, dim=-1, keepdim=True)
        std = torch.std(mag, dim=-1, keepdim=True)

        # Extremely conservative thresholds
        thresh_n_mult_map = {
            'low': 1.2,      # Bare minimum noise reduction
            'medium': 1.1,   # Very light noise reduction
            'high': 1.05,    # Light noise reduction
            'extreme': 1.0   # Standard noise reduction
        }
        thresh_n_mult = thresh_n_mult_map.get(intensity, 1.1)

        # Only target persistent background noise
        thresh = noise_floor * thresh_n_mult

        # Create an adaptive mask based on signal-to-noise ratio
        snr = mag / (noise_floor + 1e-8)
        mask = 1.0 - (1.0 / (1.0 + torch.exp(2 * (snr - thresh_n_mult))))

        # Apply mask with extremely high preservation of original signal
        blend = 0.98  # Keep 98% of original signal
        mag_cleaned = mag * (1.0 - (1.0 - blend) * mask)

        # Reconstruct with original phase
        stft_cleaned = torch.polar(mag_cleaned, phase)
        y_cleaned = torch.istft(stft_cleaned,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               window=window,
                               length=y_tensor.size(-1))

        return y_cleaned.squeeze()

    except Exception as e:
        logger.error(f"Error in spectral gating: {str(e)}")
        return y_tensor.squeeze()

def batch_process_audio(input_files, output_folder, hiss_reduction_intensity='medium'):
    """
    Process audio files with simple MP3 support
    """
    try:
        logger.info(f"Starting batch processing with intensity level: {hiss_reduction_intensity}")
        results = []

        for input_file in input_files:
            try:
                logger.info(f"Processing file: {input_file}")

                # Create a temporary directory for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Convert MP3 to WAV if needed
                    file_ext = os.path.splitext(input_file)[1].lower()
                    if file_ext == '.mp3':
                        logger.info("Converting MP3 to WAV for processing")
                        audio = AudioSegment.from_mp3(input_file)
                        temp_wav = os.path.join(temp_dir, "temp.wav")
                        audio.export(temp_wav, format="wav")
                        processing_file = temp_wav
                    else:
                        processing_file = input_file

                    # Load and process audio file
                    logger.info(f"Loading audio file: {processing_file}")
                    y, sr = torchaudio.load(processing_file)
                    logger.info(f"Loaded audio with shape {y.shape} and sample rate {sr}")

                    # Convert stereo to mono if necessary
                    if y.size(0) == 2:
                        y = y.mean(0, keepdim=True)
                        logger.info("Converted stereo to mono")

                    # Minimal normalization to preserve dynamics
                    max_val = torch.max(torch.abs(y))
                    if max_val > 1e-6:
                        y = y / max_val

                    # Process audio
                    y_processed = spectral_gating(y, sr, intensity=hiss_reduction_intensity)

                    # Restore original volume level
                    if max_val > 1e-6:
                        y_processed = y_processed * max_val

                    # Generate output filename
                    base_name = os.path.splitext(os.path.basename(input_file))[0]
                    output_filename = os.path.join(
                        output_folder,
                        f"processed_{base_name}.wav"
                    )

                    # Save processed audio
                    logger.info(f"Saving processed audio to: {output_filename}")
                    torchaudio.save(output_filename, y_processed.unsqueeze(0), sr)
                    logger.info(f"Successfully saved processed file: {output_filename}")

                    results.append({
                        'input': input_file,
                        'output': os.path.basename(output_filename),
                        'status': 'success',
                        'intensity': hiss_reduction_intensity
                    })

            except Exception as e:
                logger.error(f"Error processing {input_file}: {str(e)}")
                results.append({
                    'input': input_file,
                    'status': 'error',
                    'message': str(e)
                })

        return results

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise

if __name__ == "__main__":
    pass