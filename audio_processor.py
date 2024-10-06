import os
import numpy as np
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

def spectral_gate(mag, threshold_db):
    threshold = librosa.db_to_amplitude(threshold_db)
    mask = mag > threshold
    return mask * mag

def process_audio_chunk(chunk, sr):
    # Perform spectral gating noise reduction
    stft = librosa.stft(chunk)
    mag, phase = librosa.magphase(stft)
    
    # Apply spectral gating
    mag_db = librosa.amplitude_to_db(mag)
    mask = spectral_gate(mag_db, threshold_db=-20)
    mag_reduced = librosa.db_to_amplitude(mask)
    
    # Reconstruct the signal
    stft_reconstructed = mag_reduced * phase
    chunk_reconstructed = librosa.istft(stft_reconstructed)
    
    # Normalize audio
    chunk_reconstructed = librosa.util.normalize(chunk_reconstructed)
    
    return chunk_reconstructed

def process_audio(input_file):
    # Load the audio file using librosa
    y, sr = librosa.load(input_file)
    
    # Split audio into chunks for parallel processing
    chunk_length = sr * 5  # 5-second chunks
    chunks = [y[i:i+chunk_length] for i in range(0, len(y), chunk_length)]
    
    # Process chunks in parallel
    with ThreadPoolExecutor() as executor:
        processed_chunks = list(executor.map(lambda chunk: process_audio_chunk(chunk, sr), chunks))
    
    # Concatenate processed chunks
    processed_audio = np.concatenate(processed_chunks)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join('uploads', f'{base_name}_restored.wav')
    
    # Export as WAV (lossless)
    sf.write(output_file, processed_audio, sr)
    
    return output_file
