import os
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from audio_processor import process_audio
from flask_caching import Cache
from werkzeug.wsgi import FileWrapper
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
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'}), 400

    results = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                future = executor.submit(process_audio, file_path, app.config['UPLOAD_FOLDER'])
                futures.append((filename, future))
            else:
                results.append({'filename': file.filename, 'status': 'error', 'message': 'File type not allowed'})
        
        for filename, future in futures:
            try:
                output_path = future.result()
                if output_path:
                    results.append({'filename': filename, 'status': 'success', 'output_path': output_path})
                else:
                    results.append({'filename': filename, 'status': 'error', 'message': 'Audio processing failed'})
            except Exception as e:
                results.append({'filename': filename, 'status': 'error', 'message': str(e)})
    
    return jsonify(results), 200

@app.route('/download/<path:filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(filename))
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
