import os
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging
from audio_processor import batch_process_audio

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['SECRET_KEY'] = os.urandom(24)  # Add secret key for session management

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Upload directory set to: {app.config['UPLOAD_FOLDER']}")

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            logger.error("No file part in the request")
            return jsonify({'error': 'No file part'}), 400

        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        hiss_reduction_intensity = request.form.get('hiss_reduction_intensity', 'medium')
        logger.info(f"Received hiss reduction intensity: {hiss_reduction_intensity}")

        input_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                input_files.append(file_path)
                logger.info(f"Saved file: {file_path}")
            else:
                logger.error(f"File type not allowed: {file.filename}")
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400

        try:
            logger.info(f"Starting batch processing for {len(input_files)} files with intensity: {hiss_reduction_intensity}")
            results = batch_process_audio(input_files, app.config['UPLOAD_FOLDER'], hiss_reduction_intensity)
            logger.info("Batch processing completed")
            return jsonify(results)
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            # Clean up uploaded files in case of error
            for file_path in input_files:
                try:
                    os.remove(file_path)
                except:
                    pass
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(filename))
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404

        logger.info(f"Sending file: {file_path}")
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found'}), 404

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {str(e)}")
        raise