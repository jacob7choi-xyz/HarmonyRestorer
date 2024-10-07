import os
import numpy as np
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
import pyrubberband as pyrb

def spectral_gate(mag, threshold_db):
    threshold = librosa.db_to_amplitude(threshold_db)
    mask = mag > threshold
    return mask * mag

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

def denoise(y, sr):
    # Simple spectral subtraction for denoising
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    S_denoised = spectral_gate(S_db, threshold_db=-40)
    y_denoised = librosa.istft(librosa.db_to_amplitude(S_denoised) * np.exp(1j * np.angle(S)))
    return y_denoised

def process_audio_chunk(chunk, sr):
    # Perform denoising
    chunk_denoised = denoise(chunk, sr)
    
    # Apply source separation
    y_harmonic, y_percussive = source_separation(chunk_denoised, sr)
    
    # Apply pitch correction to the harmonic part
    y_harmonic_corrected = pitch_correction(y_harmonic, sr)
    
    # Combine the corrected harmonic part with the percussive part
    chunk_final = y_harmonic_corrected + y_percussive
    
    # Normalize audio
    chunk_final = librosa.util.normalize(chunk_final)
    
    return chunk_final

def process_audio(input_file, output_dir):
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
    output_file = os.path.join(output_dir, f'{base_name}_restored.wav')
    
    # Export as WAV (lossless)
    sf.write(output_file, processed_audio, sr)
    
    return output_file
