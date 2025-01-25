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
    Enhanced spectral gating with musical quality preservation
    """
    try:
        y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(0)
        if y_tensor.dim() == 2 and y_tensor.size(0) > 1:
            y_tensor = y_tensor.mean(0, keepdim=True)

        # Improved STFT parameters for better frequency resolution
        n_fft = 4096  # Increased for better frequency resolution
        hop_length = n_fft // 4
        window = torch.hann_window(n_fft).to(y_tensor.device)

        # Compute STFT with overlap
        stft = torch.stft(y_tensor,
                         n_fft=n_fft,
                         hop_length=hop_length,
                         window=window,
                         return_complex=True)

        mag = torch.abs(stft)
        phase = torch.angle(stft)

        # More conservative threshold multipliers
        thresh_n_mult = {
            'low': 3.5,      # Very gentle noise reduction
            'medium': 3.0,   # Conservative noise reduction
            'high': 2.5,     # Moderate noise reduction
            'extreme': 2.0   # More aggressive, but still preserving quality
        }.get(intensity, 3.0)

        # Compute threshold with frequency-dependent scaling
        freq_weights = torch.linspace(1.0, 1.5, mag.size(1)).unsqueeze(0).unsqueeze(-1).to(mag.device)
        mean = torch.mean(mag, dim=-1, keepdim=True)
        std = torch.std(mag, dim=-1, keepdim=True)
        thresh = mean + (thresh_n_mult * std * freq_weights)

        # Smooth mask creation with soft thresholding
        mask = torch.sigmoid((mag - thresh) * 2)

        # Apply mask with preservation of dynamics
        mag_cleaned = mag * mask

        # Reconstruct with original phase
        stft_cleaned = torch.polar(mag_cleaned, phase)

        # Inverse STFT with overlap-add
        y_cleaned = torch.istft(stft_cleaned,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              window=window,
                              length=y_tensor.size(-1))

        # Normalize output while preserving dynamics
        max_val = torch.max(torch.abs(y_cleaned))
        if max_val > 0:
            y_cleaned = y_cleaned * (torch.max(torch.abs(y_tensor)) / max_val)

        return y_cleaned.squeeze()

    except Exception as e:
        logger.error(f"Error in spectral gating: {str(e)}")
        return y_tensor.squeeze()

def batch_process_audio(input_files, output_folder, hiss_reduction_intensity='medium'):
    """
    Process audio files with memory-efficient streaming and MP3 support
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

                    # Normalize input while preserving dynamics
                    max_val = torch.max(torch.abs(y))
                    if max_val > 0:
                        y = y / max_val

                    # Process audio
                    y_processed = spectral_gating(y, sr, intensity=hiss_reduction_intensity)

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