import os
from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import logging
from audio_processor import batch_process_audio
import tempfile
from pydub import AudioSegment

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['SECRET_KEY'] = os.urandom(24)

# Specify your database URI (For example, SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yourdatabase.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  
db = SQLAlchemy(app)

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Upload directory set to: {app.config['UPLOAD_FOLDER']}")

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

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

        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            input_files = []
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(temp_dir, filename)
                    file.save(file_path)
                    input_files.append(file_path)
                    logger.info(f"Saved file for processing: {file_path}")
                else:
                    logger.error(f"File type not allowed: {file.filename}")
                    return jsonify({'error': f'File type not allowed: {file.filename}'}), 400

            try:
                logger.info(f"Starting batch processing for {len(input_files)} files")
                results = batch_process_audio(input_files, app.config['UPLOAD_FOLDER'], hiss_reduction_intensity)
                logger.info("Batch processing completed successfully")
                return jsonify(results)

            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        # Get requested format and new name
        format_type = request.args.get('format', 'wav')
        new_name = request.args.get('new_name', '')

        if format_type not in ['wav', 'mp3']:
            return jsonify({'error': 'Invalid format'}), 400

        # Get the original file path (always WAV from processing)
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(filename))
        if not os.path.exists(original_file_path):
            logger.error(f"File not found: {original_file_path}")
            return jsonify({'error': 'File not found'}), 404

        # Prepare download filename
        if new_name:
            download_name = f"{new_name}.{format_type}"
        else:
            download_name = os.path.splitext(filename)[0] + f".{format_type}"

        if format_type == 'wav':
            logger.info(f"Sending WAV file: {original_file_path}")
            return send_file(original_file_path, as_attachment=True, download_name=download_name)
        else:  # MP3
            # Create a temporary MP3 file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                temp_mp3_path = temp_mp3.name
                # Convert WAV to MP3
                audio = AudioSegment.from_wav(original_file_path)
                audio.export(temp_mp3_path, format='mp3', bitrate='320k')
                logger.info(f"Converted and sending MP3 file")
                return_value = send_file(temp_mp3_path, as_attachment=True, download_name=download_name)
                # Clean up temp file after sending
                os.unlink(temp_mp3_path)
                return return_value

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stream/<path:filename>')
def stream_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(filename))
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404

        logger.info(f"Streaming file: {file_path}")
        return send_file(file_path)
    except Exception as e:
        logger.error(f"Error streaming file: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5001, debug=True)  # Add this line to start the server
    except Exception as e:
        logger.error(f"Failed to start Flask server: {str(e)}")
        raise

    