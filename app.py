import os
import logging
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from audio_processor import batch_process_audio
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import secrets
from flask_migrate import Migrate
from models import db, User, Restoration

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///harmonyrestorer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg', 'flac'}

db.init_app(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists.')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
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
        logger.info(f"Starting batch processing for {len(input_files)} files")
        results = batch_process_audio(input_files, app.config['UPLOAD_FOLDER'], hiss_reduction_intensity)
        logger.info("Batch processing completed")
        
        # Save restoration history
        for result in results:
            if result['status'] == 'success':
                restoration = Restoration(
                    original_filename=os.path.basename(result['input']),
                    restored_filename=os.path.basename(result['output']),
                    user_id=current_user.id
                )
                db.session.add(restoration)
        
        db.session.commit()
        
        return jsonify(results), 200
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(filename))
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404
    
    logger.info(f"Sending file: {file_path}")
    return send_file(file_path, mimetype='audio/wav')

@app.route('/history')
@login_required
def history():
    restorations = Restoration.query.filter_by(user_id=current_user.id).order_by(Restoration.timestamp.desc()).all()
    return render_template('history.html', restorations=restorations)

@app.route('/api/restore', methods=['POST'])
def api_restore():
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return jsonify({'error': 'API key is missing'}), 401

    user = User.query.filter_by(api_key=api_key).first()
    if not user:
        return jsonify({'error': 'Invalid API key'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        hiss_reduction_intensity = request.form.get('hiss_reduction_intensity', 'medium')

        try:
            results = batch_process_audio([file_path], app.config['UPLOAD_FOLDER'], hiss_reduction_intensity)
            
            if results[0]['status'] == 'success':
                restoration = Restoration(
                    original_filename=filename,
                    restored_filename=os.path.basename(results[0]['output']),
                    user_id=user.id
                )
                db.session.add(restoration)
                db.session.commit()

            return jsonify(results[0]), 200
        except Exception as e:
            logger.error(f"Error in API restoration: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/history', methods=['GET'])
def api_history():
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return jsonify({'error': 'API key is missing'}), 401

    user = User.query.filter_by(api_key=api_key).first()
    if not user:
        return jsonify({'error': 'Invalid API key'}), 401

    restorations = Restoration.query.filter_by(user_id=user.id).order_by(Restoration.timestamp.desc()).all()
    history_data = [{
        'id': r.id,
        'original_filename': r.original_filename,
        'restored_filename': r.restored_filename,
        'timestamp': r.timestamp.isoformat()
    } for r in restorations]

    return jsonify(history_data), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)