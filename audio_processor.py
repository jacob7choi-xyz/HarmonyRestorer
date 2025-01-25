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
    Optimized spectral gating with better memory handling
    """
    try:
        y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(0)
        if y_tensor.dim() == 2 and y_tensor.size(0) > 1:
            y_tensor = y_tensor.mean(0, keepdim=True)

        n_fft = 2048
        hop_length = n_fft // 4

        stft = torch.stft(y_tensor,
                         n_fft=n_fft,
                         hop_length=hop_length,
                         window=torch.hann_window(n_fft).to(y_tensor.device),
                         return_complex=True)

        mag = torch.abs(stft)
        phase = torch.angle(stft)

        thresh_n_mult = {
            'low': 2.5,
            'medium': 2.0,
            'high': 1.5,
            'extreme': 1.0
        }.get(intensity, 2.0)

        mean = torch.mean(mag, dim=-1, keepdim=True)
        std = torch.std(mag, dim=-1, keepdim=True)
        thresh = mean + thresh_n_mult * std

        mask = (mag > thresh).float()
        mag_cleaned = mag * mask
        stft_cleaned = torch.polar(mag_cleaned, phase)

        y_cleaned = torch.istft(stft_cleaned,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              window=torch.hann_window(n_fft).to(y_tensor.device),
                              length=y_tensor.size(-1))

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

                    # Normalize input
                    y = y / (torch.max(torch.abs(y)) + 1e-6)

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