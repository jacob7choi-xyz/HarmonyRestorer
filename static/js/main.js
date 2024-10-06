document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const uploadButton = document.getElementById('upload-button');
    const audioPlayer = document.getElementById('audio-player');
    const downloadLink = document.getElementById('download-link');
    const errorMessage = document.getElementById('error-message');

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileName.textContent = file.name;
            uploadButton.disabled = false;
        } else {
            fileName.textContent = '';
            uploadButton.disabled = true;
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);

        try {
            uploadButton.disabled = true;
            uploadButton.textContent = 'Processing...';
            errorMessage.textContent = '';

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const result = await response.json();
            audioPlayer.src = result.file;
            audioPlayer.style.display = 'block';
            downloadLink.href = `/download/${encodeURIComponent(result.file)}`;
            downloadLink.style.display = 'inline-block';
        } catch (error) {
            console.error('Error:', error);
            errorMessage.textContent = 'An error occurred while processing the audio file.';
        } finally {
            uploadButton.disabled = false;
            uploadButton.textContent = 'Upload';
        }
    });
});
