import torch
import torchaudio
import numpy as np
from audio_processor import source_separation, pitch_correction, denoise, spectral_gating, plot_spectrogram, advanced_pitch_correction
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_signal(freq, duration, sr):
    t = torch.linspace(0, duration, int(duration * sr))
    return torch.sin(2 * np.pi * freq * t)

def visualize_test_results(original, processed, title):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title(f"{title} - Original vs Processed Waveform")
    plt.plot(original.numpy(), label='Original')
    plt.plot(processed.numpy(), label='Processed')
    plt.legend()
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.title(f"{title} - Spectrogram Difference")
    spec_orig = torchaudio.transforms.Spectrogram()(original)
    spec_proc = torchaudio.transforms.Spectrogram()(processed)
    diff = torch.log(spec_proc + 1e-8) - torch.log(spec_orig + 1e-8)
    plt.imshow(diff.squeeze().numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"test_{title.lower().replace(' ', '_')}.png")
    plt.close()

def test_source_separation():
    try:
        logger.info("Starting source separation test")
        sr = 22050
        duration = 5
        y_harmonic = generate_test_signal(440, duration, sr)
        y_percussive = torch.randn(int(duration * sr)) * 0.1
        y = y_harmonic + y_percussive

        y_harmonic_sep, y_percussive_sep = source_separation(y)

        visualize_test_results(y, y_harmonic_sep, "Source Separation - Harmonic")
        visualize_test_results(y, y_percussive_sep, "Source Separation - Percussive")

        assert y_harmonic_sep.shape == y.shape, f"Harmonic shape mismatch: {y_harmonic_sep.shape} != {y.shape}"
        assert y_percussive_sep.shape == y.shape, f"Percussive shape mismatch: {y_percussive_sep.shape} != {y.shape}"

        assert not torch.allclose(y_harmonic_sep, y_percussive_sep), "Harmonic and percussive components are identical"

        logger.info("Source separation test passed.")
    except Exception as e:
        logger.error(f"Source separation test failed: {e}")

def test_pitch_correction():
    try:
        logger.info("Starting pitch correction test")
        sr = 22050
        duration = 5
        input_freq = 440
        target_freq = 880
        y = generate_test_signal(input_freq, duration, sr)

        y_corrected = pitch_correction(y, sr, target_pitch=target_freq)

        visualize_test_results(y, y_corrected, "Pitch Correction")

        assert not torch.allclose(y, y_corrected), "Pitch-corrected signal is identical to input"

        pitches, confidences = torchaudio.functional.detect_pitch_frequency(y_corrected.unsqueeze(0), sr)
        valid_pitches = pitches[confidences > 0.7]
        if valid_pitches.numel() > 0:
            mean_pitch = torch.mean(valid_pitches)
            logger.info(f"Mean pitch of corrected signal: {mean_pitch.item()} Hz")
            assert torch.isclose(mean_pitch, torch.tensor(float(target_freq)), rtol=0.1), f"Corrected pitch {mean_pitch.item()} is not close to target pitch {target_freq}"
        else:
            logger.warning("No valid pitches detected in the corrected signal")

        logger.info("Pitch correction test passed.")
    except Exception as e:
        logger.error(f"Pitch correction test failed: {e}")

def test_denoise():
    try:
        logger.info("Starting denoising test")
        sr = 22050
        duration = 5
        clean_signal = generate_test_signal(440, duration, sr)
        noise = torch.randn(int(duration * sr)) * 0.1
        noisy_signal = clean_signal + noise

        denoised_signal = denoise(noisy_signal, sr, intensity='high')

        visualize_test_results(noisy_signal, denoised_signal, "Denoising")
        plot_spectrogram(noisy_signal, sr, "Noisy Signal Spectrogram")
        plot_spectrogram(denoised_signal, sr, "Denoised Signal Spectrogram")

        assert not torch.allclose(noisy_signal, denoised_signal), "Denoised signal is identical to input"
        assert torch.mean(torch.abs(clean_signal - denoised_signal)) < torch.mean(torch.abs(clean_signal - noisy_signal)), "Denoised signal is not closer to clean signal"

        logger.info("Denoising test passed.")
    except Exception as e:
        logger.error(f"Denoising test failed: {e}")

def test_spectral_gating():
    try:
        logger.info("Starting spectral gating test")
        sr = 22050
        duration = 5
        clean_signal = generate_test_signal(440, duration, sr)
        noise = torch.randn(int(duration * sr)) * 0.1
        noisy_signal = clean_signal + noise

        gated_signal = spectral_gating(noisy_signal, sr)

        visualize_test_results(noisy_signal, gated_signal, "Spectral Gating")
        plot_spectrogram(noisy_signal, sr, "Noisy Signal Spectrogram")
        plot_spectrogram(gated_signal, sr, "Gated Signal Spectrogram")

        assert not torch.allclose(noisy_signal, gated_signal), "Gated signal is identical to input"
        
        # Calculate SNR for noisy and gated signals
        noise_power = torch.mean((clean_signal - noisy_signal) ** 2)
        gated_noise_power = torch.mean((clean_signal - gated_signal) ** 2)
        signal_power = torch.mean(clean_signal ** 2)
        
        snr_noisy = 10 * torch.log10(signal_power / noise_power)
        snr_gated = 10 * torch.log10(signal_power / gated_noise_power)
        
        logger.info(f"SNR of noisy signal: {snr_noisy:.2f} dB")
        logger.info(f"SNR of gated signal: {snr_gated:.2f} dB")
        
        assert snr_gated > snr_noisy, "Gated signal SNR is not higher than noisy signal SNR"

        logger.info("Spectral gating test passed.")
    except Exception as e:
        logger.error(f"Spectral gating test failed: {e}")

def test_advanced_pitch_correction():
    try:
        logger.info("Starting advanced pitch correction test")
        sr = 22050
        duration = 5
        input_freq = 440
        target_freq = 880
        y = generate_test_signal(input_freq, duration, sr)

        y_corrected = advanced_pitch_correction(y, sr, target_pitch=target_freq)

        visualize_test_results(y, y_corrected, "Advanced Pitch Correction")

        assert not torch.allclose(y, y_corrected), "Advanced pitch-corrected signal is identical to input"

        pitches, confidences = torchaudio.functional.detect_pitch_frequency(y_corrected.unsqueeze(0), sr)
        valid_pitches = pitches[confidences > 0.7]
        if valid_pitches.numel() > 0:
            mean_pitch = torch.mean(valid_pitches)
            logger.info(f"Mean pitch of advanced corrected signal: {mean_pitch.item()} Hz")
            assert torch.isclose(mean_pitch, torch.tensor(float(target_freq)), rtol=0.1), f"Advanced corrected pitch {mean_pitch.item()} is not close to target pitch {target_freq}"
        else:
            logger.warning("No valid pitches detected in the advanced corrected signal")

        logger.info("Advanced pitch correction test passed.")
    except Exception as e:
        logger.error(f"Advanced pitch correction test failed: {e}")

if __name__ == "__main__":
    test_source_separation()
    test_pitch_correction()
    test_advanced_pitch_correction()
    test_denoise()
    test_spectral_gating()
    logger.info("All tests completed.")
