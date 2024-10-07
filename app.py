import os
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from audio_processor import process_audio
from flask_caching import Cache
from werkzeug.wsgi import FileWrapper
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
async def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    processed_files = []
    
    async def process_file(file):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            await asyncio.to_thread(file.save, file_path)
            
            try:
                # Process the audio file
                output_path = await asyncio.to_thread(process_audio, file_path)
                return output_path
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                return None
        return None

    with ThreadPoolExecutor() as executor:
        tasks = [asyncio.to_thread(process_file, file) for file in files]
        results = await asyncio.gather(*tasks)
    
    processed_files = [result for result in results if result is not None]
    
    if not processed_files:
        return jsonify({'error': 'No files were successfully processed'}), 500
    
    return jsonify({'message': 'Files uploaded and processed successfully', 'files': processed_files}), 200

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
