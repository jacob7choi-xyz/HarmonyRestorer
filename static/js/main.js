document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const uploadButton = document.getElementById('upload-button');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const errorMessage = document.getElementById('error-message');
    const resultsContainer = document.getElementById('results-container');
    const dropArea = document.getElementById('drop-area');
    let wavesurfers = {};  // Store WaveSurfer instances

    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            uploadFiles();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            updateFileName();
            toggleUploadButton();
        });
    }

    if (dropArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });

        dropArea.addEventListener('drop', handleDrop, false);
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropArea.classList.add('dragover');
    }

    function unhighlight() {
        dropArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        updateFileName();
        toggleUploadButton();
    }

    function updateFileName() {
        const files = fileInput.files;
        if (files.length > 0) {
            fileName.textContent = Array.from(files).map(file => file.name).join(', ');
        } else {
            fileName.textContent = '';
        }
    }

    function toggleUploadButton() {
        uploadButton.disabled = fileInput.files.length === 0;
    }

    async function uploadFiles() {
        const formData = new FormData(uploadForm);

        try {
            progressContainer.style.display = 'block';
            errorMessage.textContent = '';
            resultsContainer.innerHTML = '';
            uploadButton.disabled = true;

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            displayResults(data);
        } catch (error) {
            errorMessage.textContent = 'An error occurred during upload: ' + error.message;
        } finally {
            progressContainer.style.display = 'none';
            uploadButton.disabled = false;
        }
    }

    function displayResults(results) {
        results.forEach((result, index) => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';

            if (result.status === 'success') {
                const wavesurferId = `waveform-${index}`;
                resultItem.innerHTML = `
                    <h3>File: ${result.input}</h3>
                    <p>Status: Success</p>
                    <p>Processing Intensity: ${result.intensity}</p>
                    <div class="audio-controls">
                        <button class="play-pause-btn" data-file="${result.output}">
                            <i class="fas fa-play"></i>
                        </button>
                        <div id="${wavesurferId}" class="waveform"></div>
                    </div>
                    <div class="download-options">
                        <a href="/download/${encodeURIComponent(result.output)}?format=wav" class="download-link">
                            <i class="fas fa-download"></i> Download WAV
                        </a>
                        <a href="/download/${encodeURIComponent(result.output)}?format=mp3" class="download-link">
                            <i class="fas fa-download"></i> Download MP3
                        </a>
                    </div>
                `;

                resultsContainer.appendChild(resultItem);

                // Initialize WaveSurfer
                const wavesurfer = WaveSurfer.create({
                    container: `#${wavesurferId}`,
                    waveColor: '#3498db',
                    progressColor: '#2980b9',
                    cursorColor: '#2c3e50',
                    height: 80,
                    responsive: true,
                    normalize: true
                });

                // Load audio file
                wavesurfer.load(`/stream/${encodeURIComponent(result.output)}`);
                wavesurfers[result.output] = wavesurfer;

                // Add play/pause functionality
                const playPauseBtn = resultItem.querySelector('.play-pause-btn');
                playPauseBtn.addEventListener('click', function() {
                    const filename = this.dataset.file;
                    const ws = wavesurfers[filename];
                    const icon = this.querySelector('i');

                    if (ws.isPlaying()) {
                        ws.pause();
                        icon.className = 'fas fa-play';
                    } else {
                        // Pause all other players first
                        Object.values(wavesurfers).forEach(w => {
                            if (w !== ws && w.isPlaying()) {
                                w.pause();
                                const otherBtn = document.querySelector(`.play-pause-btn[data-file="${filename}"] i`);
                                if (otherBtn) otherBtn.className = 'fas fa-play';
                            }
                        });
                        ws.play();
                        icon.className = 'fas fa-pause';
                    }
                });

                // Update button when playback ends
                wavesurfer.on('finish', function() {
                    const btn = document.querySelector(`.play-pause-btn[data-file="${result.output}"] i`);
                    if (btn) btn.className = 'fas fa-play';
                });
            } else {
                resultItem.innerHTML = `
                    <h3>File: ${result.input}</h3>
                    <p>Status: Error</p>
                    <p class="error-message">Error: ${result.message}</p>
                `;
            }

            resultsContainer.appendChild(resultItem);
        });
    }
});