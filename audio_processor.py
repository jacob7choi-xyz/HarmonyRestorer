import os
from pydub import AudioSegment
import numpy as np

def process_audio(input_file):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())

    # Perform simple noise reduction (remove very quiet sounds)
    threshold = np.percentile(np.abs(samples), 5)
    samples[np.abs(samples) < threshold] = 0

    # Normalize audio
    samples = np.int16(samples / np.max(np.abs(samples)) * 32767)

    # Convert back to AudioSegment
    processed_audio = AudioSegment(
        samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join('uploads', f'{base_name}_restored.mp3')

    # Export as MP3
    processed_audio.export(output_file, format="mp3")

    return output_file
