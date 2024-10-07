document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const uploadButton = document.getElementById('upload-button');
    const progressContainer = document.getElementById('progress-container');
    const errorMessage = document.getElementById('error-message');
    const downloadLink = document.getElementById('download-link');
    const resultsContainer = document.getElementById('results-container');

    let wavesurfers = {};

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
            fileName.textContent = Array.from(files).map(file => file.name).join(', ');
            uploadButton.disabled = false;
        } else {
            fileName.textContent = '';
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
            resultsContainer.innerHTML = '';

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Upload failed');
            }

            const results = await response.json();
            displayResults(results);
        } catch (error) {
            console.error('Error:', error);
            errorMessage.textContent = error.message || 'An error occurred while processing the audio files. Please try again.';
        } finally {
            uploadButton.disabled = false;
            uploadButton.textContent = 'Upload & Restore';
            progressContainer.style.display = 'none';
        }
    });

    function displayResults(results) {
        results.forEach((result, index) => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result-item';
            resultDiv.innerHTML = `
                <h3>${result.filename}</h3>
                <p>Status: ${result.status}</p>
                ${result.status === 'success' 
                    ? `<div id="waveform-${index}"></div>
                       <button class="play-pause" data-index="${index}">Play/Pause</button>
                       <a href="/download/${encodeURIComponent(result.output_path)}" class="download-link">Download Restored Audio</a>`
                    : `<p>Error: ${result.message}</p>`
                }
            `;
            resultsContainer.appendChild(resultDiv);

            if (result.status === 'success') {
                wavesurfers[index] = initWaveSurfer(`waveform-${index}`);
                wavesurfers[index].load(result.output_path);
            }
        });

        // Add event listeners for play/pause buttons
        document.querySelectorAll('.play-pause').forEach(button => {
            button.addEventListener('click', (e) => {
                const index = e.target.getAttribute('data-index');
                wavesurfers[index].playPause();
            });
        });
    }
});
