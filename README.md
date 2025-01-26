# HarmonyRestorer 🎶

**HarmonyRestorer** is a sophisticated web-based audio restoration application designed to improve and enhance audio quality. Built with **Flask**, **PyTorch**, and **Pydub**, this tool employs advanced signal processing techniques and machine learning models to reduce unwanted noise, eliminate hiss, and enhance clarity. HarmonyRestorer offers batch processing capabilities, real-time audio previews, customizable restoration settings, and a modern web interface, making it a versatile tool for both professional audio engineers and casual users.

## Features

### Audio Restoration & Enhancement
- **Multi-Level Spectral Gating Noise Reduction**: Employs **PyTorch** to reduce unwanted background noise (e.g., hiss, hum, and static) at various intensity levels, from low to extreme, while preserving the integrity of the original audio.
- **Intelligent Stereo-to-Mono Conversion**: Automatically converts stereo recordings into mono, maintaining the natural sound quality of the original audio.
- **Comprehensive Format Support**: Process a wide range of audio formats, including MP3, WAV, OGG, and FLAC, allowing for flexibility and ease of use.
- **Batch Processing**: Efficiently handles multiple audio files simultaneously, saving time when working with large collections of audio files.
- **Real-time Audio Preview**: Provides users with the ability to hear a preview of the audio as they adjust the restoration settings, ensuring the desired outcome before finalizing the restoration.

### User-Friendly Web Interface
- **Drag-and-Drop File Upload**: Effortlessly upload audio files directly from your computer for processing.
- **Interactive Waveform Visualization**: Visualize the waveform of your audio file and monitor the restoration progress in real time.
- **Real-time Feedback**: Receive immediate visual and audio feedback while adjusting the restoration settings, enabling precise control over the restoration process.
- **Custom File Naming & Format Selection**: Choose preferred output formats (WAV or MP3) and specify custom file names for restored audio files.

### Security & Privacy
- **Secure File Handling**: Audio files are securely stored during processing and are automatically deleted once the restoration process is completed to ensure user privacy.
- **Configurable File Size Limits**: The app supports configurable file size limits (32MB by default), preventing large files from overwhelming the system.

## Technologies Used
- **Flask**: A lightweight and powerful Python web framework used for building the backend of the application, handling HTTP requests and serving dynamic content.
- **PyTorch**: Utilized for advanced audio processing and machine learning-based restoration techniques, including the implementation of **spectral gating** for noise reduction.
- **Pydub**: A robust audio processing library for converting, manipulating, and exporting audio files in various formats.
- **Werkzeug**: A comprehensive WSGI web application library used by Flask, providing essential utilities for HTTP request handling.
- **Jinja2**: A template engine for rendering HTML content dynamically in the web interface, enabling the app to serve customized user content.
- **SQLAlchemy**: A powerful database toolkit for managing user-related settings and metadata, allowing for the storage and retrieval of data within the application.

## Roadmap
- **User Authentication System**: Introduce user accounts and authentication mechanisms to allow users to save their restoration settings and access processing history.
- **Advanced Audio Restoration Algorithms**: Incorporate more complex restoration techniques, such as deep learning-based noise reduction and audio super-resolution, to further enhance audio quality.
- **Expanded Audio Format Support**: Extend the list of supported audio formats for processing, providing users with even more flexibility when working with different file types.

## Installation

To run **HarmonyRestorer** locally, follow the steps below:

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/HarmonyRestorer.git
cd HarmonyRestorer

### 2. Set up a virtual environment:
```bash
python -m venv harmonyenv
source harmonyenv/bin/activate  # On MacOS/Linux
.\harmonyenv\Scripts\activate  # On Windows
pip install -r requirements.txt

### 3. Run the application:
python app.py

Then, visit visit http://localhost:5001 to access the app locally!



## Contributing

We welcome contributions to **HarmonyRestorer**! If you'd like to contribute, please follow these steps:

1. **Fork this repository.**
2. **Clone your forked repository to your local machine.**
3. **Create a new branch for your feature or fix:**
   ```bash
   git checkout -b feature/your-feature-name
4. **Make your changes and commit them:
   ```bash
   git commit -am 'Add new feature or fix'
5. **Push your changes to your fork:
   git push origin feature/your-feature-name
6. **Create a pull request with a detailed description of your changes and why they are needed.

Please ensure that your code adheres to the existing style guidelines and includes any necessary tests!
