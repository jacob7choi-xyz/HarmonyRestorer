import os
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from audio_processor import process_audio
from flask_caching import Cache
from werkzeug.wsgi import FileWrapper
import asyncio

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg', 'flac'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
@cache.cached(timeout=300)  # Cache for 5 minutes
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
async def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        await asyncio.to_thread(file.save, file_path)
        
        try:
            # Process the audio file
            output_path = await asyncio.to_thread(process_audio, file_path)
            return jsonify({'message': 'File uploaded and processed successfully', 'file': output_path}), 200
        except Exception as e:
            return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/download/<path:filename>')
async def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    def generate():
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                yield chunk

    response = app.response_class(
        generate(),
        mimetype='audio/wav',
        direct_passthrough=True
    )
    response.headers.set('Content-Disposition', 'attachment', filename=filename)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
