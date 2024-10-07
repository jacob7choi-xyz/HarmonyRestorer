import os
import numpy as np
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
import pyrubberband as pyrb
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def estimate_noise_profile(D, high_pass_freq=5000):
    # Estimate noise profile focusing on high frequencies
    freq_bins = librosa.fft_frequencies(sr=22050, n_fft=2048)
    high_freq_mask = freq_bins >= high_pass_freq
    high_freq_content = np.mean(np.abs(D[high_freq_mask, :]), axis=1)
    noise_profile = np.tile(high_freq_content, (D.shape[1], 1)).T
    return noise_profile

def get_threshold(intensity):
    thresholds = {
        'low': -50,
        'medium': -30,
        'high': -10
    }
    threshold = thresholds.get(intensity, -30)
    logger.debug(f"Threshold for intensity {intensity}: {threshold}")
    return threshold

def spectral_gate(D, noise_profile, threshold_db):
    threshold = librosa.db_to_amplitude(threshold_db)
    mask = np.abs(D) > (noise_profile * threshold)
    return mask * D

def reduce_hiss(y, sr, intensity='medium'):
    logger.debug(f"Reducing hiss with intensity: {intensity}")
    
    # Convert to frequency domain
    D = librosa.stft(y)
    
    # Get threshold based on intensity
    threshold = get_threshold(intensity)
    
    # Apply spectral gating multiple times for higher intensities
    iterations = 1 if intensity == 'low' else 2 if intensity == 'medium' else 3
    for i in range(iterations):
        noise_profile = estimate_noise_profile(D, high_pass_freq=5000)
        mask = spectral_gate(D, noise_profile, threshold)
        D = D * mask
        logger.debug(f"Applied spectral gating iteration {i+1} for intensity {intensity}")
    
    # Convert back to time domain
    y_reduced = librosa.istft(D)
    
    return y_reduced

def source_separation(y, sr):
    # Using librosa's harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_harmonic, y_percussive

def pitch_correction(y, sr, target_pitch=None):
    if target_pitch is None:
        # Estimate the pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = pitches[magnitudes.argmax()]
        target_pitch = librosa.hz_to_midi(pitch)
    
    # Apply pitch correction using pyrubberband
    y_shifted = pyrb.pitch_shift(y, sr, n_steps=target_pitch - librosa.hz_to_midi(librosa.estimate_tuning(y=y, sr=sr)))
    return y_shifted

def denoise(y, sr, intensity='medium'):
    logger.debug(f"Denoising with intensity: {intensity}")
    # Apply hiss reduction
    y_reduced_hiss = reduce_hiss(y, sr, intensity)
    
    # Apply additional denoising if needed
    S = librosa.stft(y_reduced_hiss)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    S_denoised = spectral_gate(S_db, np.mean(S_db, axis=1, keepdims=True), threshold_db=get_threshold(intensity))
    y_denoised = librosa.istft(librosa.db_to_amplitude(S_denoised) * np.exp(1j * np.angle(S)))
    
    return y_denoised

def process_audio_chunk(chunk, sr, hiss_reduction_intensity='medium'):
    try:
        logger.debug(f"Processing audio chunk with hiss reduction intensity: {hiss_reduction_intensity}")
        
        # Apply hiss reduction multiple times based on intensity
        iterations = 1 if hiss_reduction_intensity == 'low' else 2 if hiss_reduction_intensity == 'medium' else 3
        for i in range(iterations):
            chunk = denoise(chunk, sr, intensity=hiss_reduction_intensity)
            logger.debug(f"Applied hiss reduction iteration {i+1} for intensity {hiss_reduction_intensity}")
        
        # Apply source separation
        y_harmonic, y_percussive = source_separation(chunk, sr)
        
        # Apply pitch correction to the harmonic part
        y_harmonic_corrected = pitch_correction(y_harmonic, sr)
        
        # Combine the corrected harmonic part with the percussive part
        chunk_final = y_harmonic_corrected + y_percussive
        
        # Normalize audio
        chunk_final = librosa.util.normalize(chunk_final)
        
        return chunk_final
    except Exception as e:
        logger.error(f"Error in process_audio_chunk: {e}")
        return chunk

def process_audio(input_file, output_dir, hiss_reduction_intensity='medium'):
    try:
        logger.info(f"Processing audio file: {input_file} with hiss reduction intensity: {hiss_reduction_intensity}")
        # Load the audio file using librosa
        y, sr = librosa.load(input_file)
        
        # Split audio into chunks for parallel processing
        chunk_length = sr * 5  # 5-second chunks
        chunks = [y[i:i+chunk_length] for i in range(0, len(y), chunk_length)]
        
        # Process chunks in parallel
        with ThreadPoolExecutor() as executor:
            processed_chunks = list(executor.map(lambda chunk: process_audio_chunk(chunk, sr, hiss_reduction_intensity), chunks))
        
        # Concatenate processed chunks
        processed_audio = np.concatenate(processed_chunks)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f'{base_name}_restored.wav')
        
        # Export as WAV (lossless)
        sf.write(output_file, processed_audio, sr)
        
        logger.info(f"Audio processing completed. Output file: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error processing audio file {input_file}: {e}")
        return None

def batch_process_audio(input_files, output_dir, hiss_reduction_intensity='medium'):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_audio, input_file, output_dir, hiss_reduction_intensity) for input_file in input_files]
        for future, input_file in zip(futures, input_files):
            try:
                output_file = future.result()
                if output_file:
                    results.append({'input': input_file, 'output': output_file, 'status': 'success'})
                else:
                    results.append({'input': input_file, 'status': 'error', 'message': 'Processing failed'})
            except Exception as e:
                logger.error(f"Error in batch processing for file {input_file}: {e}")
                results.append({'input': input_file, 'status': 'error', 'message': str(e)})
    return results
