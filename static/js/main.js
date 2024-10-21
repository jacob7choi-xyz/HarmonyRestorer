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
        if (fileInput && fileName) {
            const files = fileInput.files;
            if (files.length > 0) {
                fileName.textContent = Array.from(files).map(file => file.name).join(', ');
            } else {
                fileName.textContent = '';
            }
        }
    }

    function toggleUploadButton() {
        if (uploadButton && fileInput) {
            uploadButton.disabled = fileInput.files.length === 0;
        }
    }

    function uploadFiles() {
        if (!uploadForm) return;
        
        const formData = new FormData(uploadForm);
        
        if (progressContainer) progressContainer.style.display = 'block';
        if (errorMessage) errorMessage.textContent = '';
        if (resultsContainer) resultsContainer.innerHTML = '';

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (progressContainer) progressContainer.style.display = 'none';
            displayResults(data);
        })
        .catch(error => {
            if (progressContainer) progressContainer.style.display = 'none';
            if (errorMessage) errorMessage.textContent = 'An error occurred during upload: ' + error.message;
        });
    }

    function displayResults(results) {
        if (!resultsContainer) return;

        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            
            if (result.status === 'success') {
                resultItem.innerHTML = `
                    <h3>File: ${result.input}</h3>
                    <p>Status: Success</p>
                    <a href="/download/${encodeURIComponent(result.output)}" class="download-link">Download Restored Audio</a>
                `;
            } else {
                resultItem.innerHTML = `
                    <h3>File: ${result.input}</h3>
                    <p>Status: Error</p>
                    <p>Message: ${result.message}</p>
                `;
            }
            
            resultsContainer.appendChild(resultItem);
        });
    }

    const togglePasswordButtons = document.querySelectorAll('.toggle-password');
    togglePasswordButtons.forEach(button => {
        if (button) {
            button.addEventListener('click', function() {
                const passwordInput = this.previousElementSibling;
                const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
                passwordInput.setAttribute('type', type);
                this.querySelector('i').classList.toggle('fa-eye');
                this.querySelector('i').classList.toggle('fa-eye-slash');
            });
        }
    });
});
