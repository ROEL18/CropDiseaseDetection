import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import uuid
from flask_cors import CORS
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# Configure application
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///plant_disease.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.config["ALLOWED_EXTENSIONS"] = {'png', 'jpg', 'jpeg', 'gif'}

# Enable CORS
CORS(app)

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Import models and initialize database
from models import db, User, PlantDiseaseResult
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()

# Import services after app creation
from plant_disease_detector import detect_disease
from gemini_service import get_treatment_recommendation, chat_with_gemini, initialize_chat

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@lru_cache(maxsize=100)
def get_cached_treatment(disease_name):
    """Cache treatment recommendations to reduce API calls"""
    return get_treatment_recommendation(disease_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    try:
        # Fetch the latest 20 results
        results = PlantDiseaseResult.query.order_by(PlantDiseaseResult.timestamp.desc()).limit(20).all()
        return render_template('history.html', results=results)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        flash("Error loading history. Please try again later.", "error")
        return redirect(url_for('index'))

@app.route('/chat')
def chat():
    try:
        # Get the disease_id from the URL parameter
        disease_id = request.args.get('disease_id')
        disease_result = None
        
        if disease_id:
            disease_result = PlantDiseaseResult.query.filter_by(id=disease_id).first()
            if not disease_result:
                flash("Disease result not found.", "error")
                return redirect(url_for('index'))
        
        return render_template('chat.html', disease_result=disease_result)
    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}")
        flash("Error loading chat. Please try again later.", "error")
        return redirect(url_for('index'))

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Generate a unique filename
        filename = secure_filename(file.filename)
        file_extension = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the uploaded file
        file.save(file_path)
        
        # Process the image with the disease detection model
        prediction, confidence = detect_disease(file_path)
        
        # Save the result to the database
        result = PlantDiseaseResult(
            image_path=file_path,
            prediction=prediction,
            confidence=float(confidence),
            timestamp=datetime.now()
        )
        db.session.add(result)
        db.session.commit()
        
        return jsonify({
            'id': result.id,
            'image_path': file_path,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': result.timestamp.isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up the file if it was saved
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_treatment', methods=['POST'])
def get_treatment():
    try:
        data = request.json
        
        if not data or 'disease' not in data:
            return jsonify({'error': 'Disease name is required'}), 400
        
        disease_name = data['disease']
        treatment = get_cached_treatment(disease_name)
        return jsonify({'treatment': treatment}), 200
    
    except Exception as e:
        logger.error(f"Error getting treatment recommendation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    try:
        # Get the latest results (limited to 20)
        results = PlantDiseaseResult.query.order_by(PlantDiseaseResult.timestamp.desc()).limit(20).all()
        
        results_list = [result.to_dict() for result in results]
        return jsonify(results_list), 200
    
    except Exception as e:
        logger.error(f"Error fetching results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/result/<int:result_id>', methods=['GET'])
def get_result(result_id):
    try:
        result = PlantDiseaseResult.query.get_or_404(result_id)
        return jsonify(result.to_dict()), 200
    except Exception as e:
        logger.error(f"Error fetching result {result_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.json
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get user message and session ID
        user_message = data['message']
        
        # Get or create a session ID for this chat
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Also allow overriding the session ID from the request
        if 'session_id' in data:
            session_id = data['session_id']
        
        # Get response from Gemini
        response = chat_with_gemini(session_id, user_message)
        
        return jsonify({
            'response': response,
            'session_id': session_id
        }), 200
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# For testing purposes only
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
