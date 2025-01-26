# HarmonyRestorer 🎶

**HarmonyRestorer** is a sophisticated web-based audio restoration application designed to enhance and restore audio quality. By leveraging cutting-edge signal processing techniques and advanced machine learning models, it reduces unwanted noise, eliminates hiss, and enhances clarity. Built with **Flask** and **PyTorch**, it offers batch processing capabilities, real-time audio previews, customizable settings, and a modern web interface, making it ideal for both professional audio engineers and casual users.

## Features

### Audio Restoration & Enhancement
- **Multi-level Spectral Gating Noise Reduction**: Reduces unwanted background noise, including hiss, hum, and static at various intensity levels (from low to extreme).
- **Intelligent Stereo-to-Mono Conversion**: Automatically converts stereo recordings into mono while preserving the integrity of the original audio.
- **Support for Multiple Audio Formats**: Easily upload and process audio formats such as MP3, WAV, OGG, and FLAC.
- **Batch Processing**: Efficiently processes multiple files at once, saving time when working with large collections.
- **Real-time Audio Preview**: Hear a preview of your audio while adjusting restoration settings.

### User-Friendly Web Interface
- **Drag-and-Drop File Upload**: Upload files directly from your computer.
- **Interactive Waveform Visualization**: View your audio file's waveform and track restoration progress.
- **Real-time Feedback**: Get immediate visual and audio feedback as you adjust restoration settings.
- **Custom File Naming & Format Selection**: Choose your preferred file name and format (WAV or MP3) for restored files.

### Security & Privacy
- **Secure File Handling**: Files are securely stored during processing and automatically deleted once the restoration is complete.
- **Configurable File Size Limits**: Prevent large files from overwhelming the system by setting adjustable file size limits (32MB).

## Technologies Used
- **Flask**: A lightweight Python web framework used for building the backend of the application.
- **Pydub**: A powerful audio processing library for converting and manipulating audio files.
- **PyTorch**: For advanced audio processing and machine learning-based restoration techniques.
- **Werkzeug**: A comprehensive WSGI web application library used by Flask.
- **Jinja2**: Template engine for rendering HTML content dynamically in the web interface.
- **SQLAlchemy**: A database toolkit for managing the app's database (if used for user settings or file metadata).

## Roadmap
- **User Account System**: Implement user accounts and authentication for saving restoration settings and processing history.
- **Advanced Restoration Algorithms**: Introduce more complex restoration techniques like deep learning-based noise reduction and audio super-resolution.
- **Support for More Audio Formats**: Expand the list of supported audio formats for processing.

## Installation

To run **HarmonyRestorer** locally on your machine, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/HarmonyRestorer.git
cd HarmonyRestorer

### 2. Set up a virtual environment:
```bash
conda create --name harmony-restorer python=3.11
conda activate harmony-restorer
pip install -r requirements.txt

### 3. Run the application:
```bash
python app.py


Contributing
We welcome contributions to HarmonyRestorer! If you'd like to contribute, feel free to fork the repository and submit a pull request. Here's how you can get started:

Fork this repository.
Create a new branch for your feature or fix (git checkout -b feature/your-feature-name).
Make your changes and commit them (git commit -am 'Add new feature').
Push to your forked repository (git push origin feature/your-feature-name).
Create a pull request with a clear description of the changes.
Please ensure that your code follows the existing style and includes tests if applicable.
