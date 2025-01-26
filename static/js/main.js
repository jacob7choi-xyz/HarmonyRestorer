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

    function showRenameDialog(originalFilename, format, downloadUrl) {
        console.log('Opening rename dialog for:', originalFilename, format, downloadUrl);
        const dialog = document.createElement('div');
        dialog.className = 'rename-dialog';

        // Remove extension from original filename
        const nameWithoutExt = originalFilename.replace(/\.[^/.]+$/, '');

        dialog.innerHTML = `
            <div class="rename-dialog-content">
                <h3>Rename File</h3>
                <input type="text" id="new-filename" value="${nameWithoutExt}" class="rename-input">
                <div class="rename-dialog-buttons">
                    <button class="cancel-btn">Cancel</button>
                    <button class="confirm-btn">Download</button>
                </div>
            </div>
        `;

        document.body.appendChild(dialog);
        console.log('Dialog created and added to body');

        const input = dialog.querySelector('#new-filename');
        const confirmBtn = dialog.querySelector('.confirm-btn');
        const cancelBtn = dialog.querySelector('.cancel-btn');

        input.focus();
        input.select();

        function closeDialog() {
            console.log('Closing dialog');
            document.body.removeChild(dialog);
        }

        confirmBtn.addEventListener('click', () => {
            const newName = input.value.trim();
            console.log('Confirming download with new name:', newName);
            if (newName) {
                const finalUrl = `${downloadUrl}&new_name=${encodeURIComponent(newName)}`;
                console.log('Final download URL:', finalUrl);
                window.location.href = finalUrl;
            }
            closeDialog();
        });

        cancelBtn.addEventListener('click', closeDialog);

        // Close dialog when clicking outside
        dialog.addEventListener('click', (e) => {
            if (e.target === dialog) {
                closeDialog();
            }
        });

        // Handle Enter key
        input.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') {
                confirmBtn.click();
            }
        });
    }

    function addDownloadButtonListeners(resultItem, result) {
        const downloadButtons = resultItem.querySelectorAll('.download-link');
        downloadButtons.forEach(btn => {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                console.log('Download button clicked');
                const format = this.dataset.format;
                const file = this.dataset.file;
                console.log('Format:', format, 'File:', file);
                const downloadUrl = `/download/${encodeURIComponent(file)}?format=${format}`;
                showRenameDialog(file, format, downloadUrl);
            });
        });
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
                        <button class="download-link" data-format="wav" data-file="${result.output}">
                            <i class="fas fa-download"></i> Download WAV
                        </button>
                        <button class="download-link" data-format="mp3" data-file="${result.output}">
                            <i class="fas fa-download"></i> Download MP3
                        </button>
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

                // Add download button listeners
                addDownloadButtonListeners(resultItem, result);
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


