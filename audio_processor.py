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
    Conservative spectral gating focused only on hiss reduction while preserving audio quality
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

        # Very conservative thresholds that mainly target high-frequency noise
        thresh_n_mult_map = {
            'low': 6.0,        # Extremely gentle
            'medium': 5.0,     # Very gentle
            'high': 4.0,       # Gentle
            'extreme': 3.0     # More noticeable but still conservative
        }
        thresh_n_mult = thresh_n_mult_map.get(intensity, 5.0)

        # Only apply significant reduction to high frequencies
        freqs = torch.linspace(0.1, 1.0, mag.size(1)).unsqueeze(0).unsqueeze(-1).to(mag.device)
        mean = torch.mean(mag, dim=-1, keepdim=True)
        std = torch.std(mag, dim=-1, keepdim=True)

        # Higher threshold for low frequencies (to preserve them) and lower for high frequencies
        thresh = mean + (thresh_n_mult * std * freqs)

        # Simple thresholding that preserves most of the signal
        mask = (mag > thresh).float()

        # Blend original and processed signals to preserve quality
        blend = 0.8  # Keep 80% of original signal
        mag_cleaned = blend * mag + (1 - blend) * (mag * mask)

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