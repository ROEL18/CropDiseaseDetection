import os
import cv2
import numpy as np
import torch
import json
import logging
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize model and processor
MODEL_PATH = "D:/project3"
processor = None
model = None

def load_model():
    """Load the trained model and processor"""
    global processor, model
    if processor is None or model is None:
        try:
            processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
            model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
            model.eval()  # Set to evaluation mode
            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def preprocess_image(image_path):
    """Preprocess the image for the model"""
    try:
        # Read the image using PIL
        image = Image.open(image_path)
        
        # Process the image using the model's processor
        inputs = processor(images=image, return_tensors="pt")
        
        return inputs
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def detect_disease(image_path):
    """Detect plant disease from an image using the trained model"""
    try:
        # Ensure model is loaded
        load_model()
        
        # Preprocess the image
        inputs = preprocess_image(image_path)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top prediction
            top_prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][top_prediction].item()
            
            # Get class name from config
            with open(os.path.join(MODEL_PATH, "config.json"), "r") as f:
                config = json.load(f)
                id2label = config.get("id2label", {})
                predicted_class = id2label.get(str(top_prediction), f"Class_{top_prediction}")
        
        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.4f}")
        return predicted_class, confidence
    
    except Exception as e:
        logger.error(f"Error detecting disease: {str(e)}")
        raise

# The class names for the plant disease detection model
CLASSES = [
    'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_Healthy',
    'Background_without_leaves', 'Blueberry_Healthy', 'Cherry_Powdery_mildew', 'Cherry_Healthy',
    'Corn_Cercospora_leaf_spot', 'Corn_Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn_Healthy',
    'Grape_Black_rot', 'Grape_Esca', 'Grape_Leaf_blight', 'Grape_Healthy',
    'Orange_Haunglongbing', 'Peach_Bacterial_spot', 'Peach_Healthy',
    'Pepper_Bacterial_spot', 'Pepper_Healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_Healthy',
    'Raspberry_Healthy', 'Soybean_Healthy', 'Squash_Powdery_mildew',
    'Strawberry_Leaf_scorch', 'Strawberry_Healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites', 'Tomato_Target_Spot', 'Tomato_Mosaic_virus', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Healthy'
]

# Common disease indicators and their corresponding colors in HSV space
DISEASE_INDICATORS = {
    "Brown spots": [(10, 100, 20), (20, 255, 200)],  # Brown
    "Yellow spots": [(20, 100, 100), (30, 255, 255)],  # Yellow
    "Black spots": [(0, 0, 0), (180, 255, 30)],  # Black
    "White powder": [(0, 0, 200), (180, 30, 255)],  # White
    "Rotting": [(0, 50, 10), (15, 255, 100)]  # Dark brown
}

def extract_color_features(img):
    """Extract color features from the image"""
    # Convert to HSV color space for better color feature extraction
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Initialize feature vector
    features = []
    
    # Check for each disease indicator (color range)
    for indicator, (lower, upper) in DISEASE_INDICATORS.items():
        # Create NumPy arrays from the boundaries
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        # Find the colors within the specified boundaries
        mask = cv2.inRange(hsv, lower, upper)
        
        # Calculate percentage of image with this color indicator
        ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])
        features.append(ratio)
    
    # Add average color features (mean of each HSV channel)
    for i in range(3):
        features.append(np.mean(hsv[:, :, i]) / 255.0)
    
    return np.array(features)

def extract_texture_features(img):
    """Extract texture features using gradient magnitude"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate gradient magnitude (simple edge detection)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize and add mean, std as features
    gradient_mag = gradient_mag / gradient_mag.max() if gradient_mag.max() > 0 else gradient_mag
    
    features = [
        np.mean(gradient_mag),  # Average edge strength
        np.std(gradient_mag),   # Variation in edge strength
        np.percentile(gradient_mag, 90)  # Strong edge percentile
    ]
    
    return np.array(features)

def simple_disease_classifier(features):
    """A simple classifier based on extracted features"""
    # This is a simplified approach that simulates the ResNet model
    # In a production environment, a properly trained model would be used
    
    # Split features
    color_features = features[:8]  # First 8 features are color-related
    texture_features = features[8:]  # Last 3 features are texture-related
    
    # Check for specific disease patterns based on feature thresholds
    
    # High level of brown spots and high edge variation (Apple scab, Black rot)
    if color_features[0] > 0.15 and texture_features[1] > 0.2:
        return 1 if random.random() > 0.5 else 0  # Apple_Black_rot or Apple_Apple_scab
    
    # High yellow spots (Leaf spot diseases)
    if color_features[1] > 0.2:
        return 30  # Tomato_Early_blight
    
    # White powdery appearance (Powdery mildew)
    if color_features[3] > 0.1:
        return 6  # Cherry_Powdery_mildew
    
    # Dark spots (Late blight)
    if color_features[2] > 0.12:
        return 31  # Tomato_Late_blight
    
    # Check if mostly green (healthy)
    if np.mean(color_features[:5]) < 0.1 and color_features[5] > 0.4:  # Low disease indicators and high green
        # Choose a random healthy class
        healthy_indices = [3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38]
        return random.choice(healthy_indices)
    
    # Fall back to a random disease if no specific pattern is detected
    # This simulates the model prediction for demonstration purposes
    return random.randint(0, len(CLASSES) - 1)

def load_sample_predictions():
    """Load sample predictions from JSON file if available"""
    sample_predictions = [
        ("Apple_Scab", 0.904),
        ("Tomato_Late_blight", 0.856),
        ("Potato_Healthy", 0.736),
        ("Grape_Black_rot", 0.892),
        ("Corn_Common_rust", 0.817)
    ]
    
    # Try to load from several possible locations
    possible_locations = [
        os.path.join('attached_assets', 'detection_results.json'),
        os.path.join('static', 'data', 'detection_results.json'),
        'detection_results.json'
    ]
    
    for detection_results_file in possible_locations:
        # Load sample predictions from detection_results.json if available
        if os.path.exists(detection_results_file):
            try:
                with open(detection_results_file, 'r') as f:
                    results = json.load(f)
                    loaded_predictions = []
                    for item in results:
                        if 'prediction' in item and 'confidence' in item:
                            loaded_predictions.append((item['prediction'], float(item['confidence'])))
                    
                    if loaded_predictions:
                        sample_predictions = loaded_predictions
                        logger.info(f"Loaded {len(sample_predictions)} sample predictions from {detection_results_file}")
                        break  # Stop looking once we've found and loaded a file
            except Exception as e:
                logger.warning(f"Failed to load sample predictions from {detection_results_file}: {str(e)}")
    
    return sample_predictions
