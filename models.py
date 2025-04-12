from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication (optional for future expansion)"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    # ensure password hash field has length of at least 256
    password_hash = db.Column(db.String(256))
    
    # Relationship with disease results
    results = db.relationship('PlantDiseaseResult', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class PlantDiseaseResult(db.Model):
    """Model to store plant disease detection results"""
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PlantDiseaseResult {self.id} - {self.prediction}>'

    def to_dict(self):
        """Convert the model instance to a dictionary"""
        return {
            'id': self.id,
            'image_path': self.image_path,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat()
        }
