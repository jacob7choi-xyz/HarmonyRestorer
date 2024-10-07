import numpy as np
import librosa
from audio_processor import source_separation, pitch_correction

def test_source_separation():
    # Generate a simple test signal
    t = np.linspace(0, 5, 5 * 22050)  # 5 seconds at 22050 Hz
    harmonic = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    percussive = np.random.randn(len(t)) * 0.1  # White noise
    y = harmonic + percussive
    sr = 22050

    # Apply source separation
    y_harmonic, y_percussive = source_separation(y, sr)

    # Check if the separated signals are different from the input
    assert not np.allclose(y, y_harmonic)
    assert not np.allclose(y, y_percussive)
    print("Source separation test passed.")

def test_pitch_correction():
    # Generate a simple test signal
    t = np.linspace(0, 5, 5 * 22050)  # 5 seconds at 22050 Hz
    y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    sr = 22050

    # Apply pitch correction
    y_corrected = pitch_correction(y, sr, target_pitch=librosa.note_to_hz('A5'))

    # Check if the corrected signal is different from the input
    assert not np.allclose(y, y_corrected)
    print("Pitch correction test passed.")

if __name__ == "__main__":
    test_source_separation()
    test_pitch_correction()
    print("All tests passed successfully!")
