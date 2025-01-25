# HarmonyRestorer 🎵

**HarmonyRestorer** is a sophisticated web-based application for **audio restoration**, designed to enhance and restore the quality of audio recordings. This app leverages cutting-edge signal processing techniques and advanced machine learning models to reduce noise, eliminate unwanted hiss, and enhance clarity, making it ideal for both professional audio engineers and casual users.

Built with **Flask** as the backend and **Pydub** for audio processing, HarmonyRestorer is capable of handling a variety of audio formats including MP3, WAV, OGG, and FLAC. It offers batch processing capabilities, real-time audio previews, and customizable settings to tailor the restoration process to user needs.

## Features

### 🎧 **Audio Restoration and Enhancement**
- **Multi-level Spectral Gating Noise Reduction**: Reduce unwanted background noise, including hiss, hum, and static at various intensity levels (from light to heavy).
- **Intelligent Stereo-to-Mono Conversion**: Automatically converts stereo recordings into mono while preserving the integrity of the original audio.
- **Support for Multiple Audio Formats**: Easily upload and process common audio formats such as MP3, WAV, OGG, and FLAC.
- **Batch Processing**: Upload and process multiple audio files at once, saving you time when working with large collections of files.
- **Real-time Audio Preview**: Hear a live preview of your audio with the restoration effects applied, allowing you to fine-tune settings.
  
### 🌐 **User-Friendly Web Interface**
- **Drag-and-Drop File Upload**: Simple and intuitive file uploads directly from your computer.
- **Interactive Waveform Visualization**: View your audio file’s waveform and track restoration progress.
- **Real-Time Feedback**: Get immediate visual and audio feedback as you adjust restoration settings.
- **Custom File Naming & Format Selection**: Choose your preferred file name and export format (WAV or MP3) for restored files.

### 🔒 **Security & Privacy**
- **Secure File Handling**: Files are temporarily stored on the server during processing and are automatically deleted after processing is complete.
- **Configurable File Size Limits**: Prevent large files from overwhelming the system with an adjustable file size limit (32MB).

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


## Technologies Used
- **Flask**: A lightweight Python web framework used for building the backend of the application.
- **Pydub**: A powerful audio processing library for converting and manipulating audio files.
- **PyTorch**: For advanced audio processing and machine learning-based restoration techniques (if applicable).
- **Werkzeug**: A comprehensive WSGI web application library used by Flask.
- **Jinja2**: Template engine for rendering HTML content dynamically in the web interface.
- **SQLAlchemy**: A database toolkit for managing the app's database (if used for user settings or file metadata).

## Roadmap
- **User Account System**: Implement user accounts and authentication for saving restoration settings and processing history.
- **Advanced Restoration Algorithms**: Introduce more complex restoration techniques like deep learning-based noise reduction and audio super-resolution.
- **Support for More Audio Formats**: Expand the list of supported audio formats for processing.

## Contributing
We welcome contributions to **HarmonyRestorer**! If you'd like to contribute, feel free to fork the repository and submit a pull request. Here's how you can get started:

- Fork this repository.
- Create a new branch for your feature or fix:
  ```bash
  git checkout -b feature/your-feature-name





### To add this to your `README.md`:
1. Copy the text above.
2. Open your `README.md` file in your project.
3. Paste the copied content into the file.
4. **Commit and push** the changes to GitHub.

Let me know if you need more assistance!
