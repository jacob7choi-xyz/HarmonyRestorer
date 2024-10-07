document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const uploadButton = document.getElementById('upload-button');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const downloadLinks = document.getElementById('download-links');
    const errorMessage = document.getElementById('error-message');
    const playPauseOriginal = document.getElementById('playPauseOriginal');
    const playPauseRestored = document.getElementById('playPauseRestored');
    const volumeSlider = document.getElementById('volume');

    let wavesurferOriginal, wavesurferRestored;

    function initWaveSurfer(containerId) {
        return WaveSurfer.create({
            container: `#${containerId}`,
            waveColor: '#4a90e2',
            progressColor: '#f5a623',
            responsive: true,
            cursorWidth: 2,
            cursorColor: '#333',
            barWidth: 2,
            barRadius: 3,
            height: 128,
            backend: 'MediaElement',
            plugins: [
                WaveSurfer.cursor.create({
                    showTime: true,
                    opacity: 1,
                    customShowTimeStyle: {
                        'background-color': '#000',
                        color: '#fff',
                        padding: '2px',
                        'font-size': '10px'
                    }
                })
            ],
        });
    }

    wavesurferOriginal = initWaveSurfer('waveform-original');
    wavesurferRestored = initWaveSurfer('waveform-restored');

    playPauseOriginal.addEventListener('click', () => {
        wavesurferOriginal.playPause();
        playPauseOriginal.textContent = wavesurferOriginal.isPlaying() ? 'Pause Original' : 'Play Original';
    });

    playPauseRestored.addEventListener('click', () => {
        wavesurferRestored.playPause();
        playPauseRestored.textContent = wavesurferRestored.isPlaying() ? 'Pause Restored' : 'Play Restored';
    });

    volumeSlider.addEventListener('input', (e) => {
        const volume = parseFloat(e.target.value);
        wavesurferOriginal.setVolume(volume);
        wavesurferRestored.setVolume(volume);
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('dragover');
    }

    function unhighlight() {
        dropArea.classList.remove('dragover');
    }

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            fileList.innerHTML = '';
            Array.from(files).forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.textContent = file.name;
                fileList.appendChild(fileItem);
            });
            uploadButton.disabled = false;
        } else {
            fileList.innerHTML = '';
            uploadButton.disabled = true;
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);

        try {
            uploadButton.disabled = true;
            uploadButton.textContent = 'Processing...';
            errorMessage.textContent = '';
            progressContainer.style.display = 'block';
            downloadLinks.innerHTML = '';

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Upload failed');
            }

            const result = await response.json();
            result.files.forEach(file => {
                const link = document.createElement('a');
                link.href = `/download/${encodeURIComponent(file)}`;
                link.textContent = `Download ${file}`;
                link.className = 'download-link';
                downloadLinks.appendChild(link);
            });

            // Load the first restored file for preview
            if (result.files.length > 0) {
                wavesurferRestored.load(result.files[0]);
            }
        } catch (error) {
            console.error('Error:', error);
            errorMessage.textContent = error.message || 'An error occurred while processing the audio files. Please try again.';
        } finally {
            uploadButton.disabled = false;
            uploadButton.textContent = 'Upload & Restore';
            progressContainer.style.display = 'none';
        }
    });

    function updateProgress(progress) {
        progressBar.style.width = `${progress}%`;
    }

    let progressInterval;
    form.addEventListener('submit', () => {
        let progress = 0;
        progressInterval = setInterval(() => {
            progress += 10;
            if (progress > 100) {
                clearInterval(progressInterval);
            } else {
                updateProgress(progress);
            }
        }, 500);
    });
});
