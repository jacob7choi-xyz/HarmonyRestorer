import os
import logging
import numpy as np
import soundfile as sf
from audio_processor import batch_process_audio

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='test_batch_processing.log',
                    filemode='w')
logger = logging.getLogger(__name__)

def create_test_audio(file_path, duration=5, sample_rate=22050):
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    audio = np.sin(440 * 2 * np.pi * t) * 0.3  # 440 Hz sine wave
    sf.write(file_path, audio, sample_rate)
    logger.info(f"Created test audio file: {file_path}")

def test_batch_processing():
    logger.info("Starting batch processing test")
    
    # Create a test directory for input and output files
    test_dir = os.path.join(os.getcwd(), 'test_files')
    os.makedirs(test_dir, exist_ok=True)
    logger.info(f"Created test directory: {test_dir}")

    # Create test input files
    input_files = [
        os.path.join(test_dir, 'test_audio1.wav'),
        os.path.join(test_dir, 'test_audio2.wav'),
        os.path.join(test_dir, 'test_audio3.wav')
    ]

    # Create actual audio files for testing
    for file in input_files:
        create_test_audio(file)

    # Test batch processing with different intensities
    intensities = ['low', 'medium', 'high', 'extreme']

    for intensity in intensities:
        logger.info(f"Testing batch processing with {intensity} intensity")
        try:
            results = batch_process_audio(input_files, test_dir, hiss_reduction_intensity=intensity)
            for result in results:
                if result['status'] == 'success':
                    logger.info(f"Successfully processed {result['input']} to {result['output']}")
                    # Verify that the output file exists and has non-zero size
                    if os.path.exists(result['output']) and os.path.getsize(result['output']) > 0:
                        logger.info(f"Output file {result['output']} exists and has non-zero size")
                    else:
                        logger.error(f"Output file {result['output']} is missing or empty")
                else:
                    logger.error(f"Failed to process {result['input']}: {result['message']}")
        except Exception as e:
            logger.exception(f"Error during batch processing with {intensity} intensity: {str(e)}")

    # Clean up test files
    logger.info("Cleaning up test files")
    for file in input_files:
        os.remove(file)
        logger.info(f"Removed input file: {file}")
    for file in os.listdir(test_dir):
        if file.endswith('_restored.wav'):
            os.remove(os.path.join(test_dir, file))
            logger.info(f"Removed output file: {file}")
    os.rmdir(test_dir)
    logger.info(f"Removed test directory: {test_dir}")

if __name__ == "__main__":
    logger.info("Starting test_batch_processing.py")
    test_batch_processing()
    logger.info("Test batch processing completed")
