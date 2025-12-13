import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn import svm
import joblib
import os

class EmotionAnalyzer:
    def __init__(self, classifier_path='emotion_classifier.pkl'):
        """
        Initialize the Emotion Analysis Engine.
        Loads MTCNN for face detection and VGG16 for feature extraction.
        Attempts to load a pre-trained SVM classifier if available.
        """
        print("Loading Face Detection Model (MTCNN)...")
        self.face_detector = MTCNN()
        
        print("Loading Feature Extractor (VGG16)...")
        # Include top=False to use it as a feature extractor (removes classification layer)
        self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # Create a new model that outputs the VGG16 features
        self.feature_extractor = Model(inputs=self.base_model.input, outputs=self.base_model.output)
        
        self.classifier_path = classifier_path
        self.classifier = None
        self.load_classifier()

    def load_classifier(self):
        """Loads the trained SVM classifier from disk."""
        if os.path.exists(self.classifier_path):
            print(f"Loading emotion classifier from {self.classifier_path}...")
            self.classifier = joblib.load(self.classifier_path)
        else:
            print("No pre-trained classifier found. You will need to train the model first.")
            # Initialize a new SVM for training later
            self.classifier = svm.SVC(kernel='linear', probability=True)

    def preprocess_face(self, face_img):
        """
        Resizes and normalizes the face image for VGG16.
        """
        # Resize to VGG16 input size
        face_resized = cv2.resize(face_img, (224, 224))
        # Convert to float and normalize
        face_normalized = face_resized / 255.0
        # Add batch dimension (1, 224, 224, 3)
        face_batch = np.expand_dims(face_normalized, axis=0)
        return face_batch

    def extract_features(self, face_img):
        """
        Extracts features from the face image using VGG16.
        Returns a flattened feature vector.
        """
        preprocessed = self.preprocess_face(face_img)
        features = self.feature_extractor.predict(preprocessed, verbose=0)
        # Flatten the features: VGG16 output is (1, 7, 7, 512) -> flatten to (1, 25088) or similar depending on pooling
        # Actually VGG16 block5_pool output is 7x7x512. Flattened it is 25088.
        features_flattened = features.reshape(1, -1)
        return features_flattened

    def detect_and_classify(self, image_path_or_array):
        """
        Main pipeline function:
        1. Detect Face
        2. Extract Features
        3. Predict Emotion
        """
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                return "Error: Could not read image", 0.0, None
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)

        # Stage 1: Face Detection
        detections = self.face_detector.detect_faces(rgb_img)
        
        if not detections:
            return "No face detected", 0.0, None

        # Process the largest face found
        # Detections are a list of dicts: [{'box': [x, y, w, h], 'confidence': 0.99, ...}]
        largest_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        x, y, width, height = largest_face['box']
        
        # Ensure coordinates are within image bounds
        x, y = max(0, x), max(0, y)
        
        # Crop the face
        face_img = rgb_img[y:y+height, x:x+width]

        # Stage 2: Feature Extraction & Classification
        if not hasattr(self.classifier, 'classes_'):
             return "Model not trained", 0.0, (x, y, width, height)

        try:
            features = self.extract_features(face_img)
            
            emotion_label = self.classifier.predict(features)[0]
            confidence = np.max(self.classifier.predict_proba(features))
            
            return emotion_label, confidence, (x, y, width, height)
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error during prediction", 0.0, (x, y, width, height)

    def train(self, training_data, labels):
        """
        Trains the SVM classifier.
        training_data: list of face images (numpy arrays) or pre-extracted feature vectors.
        labels: list of emotion labels.
        """
        print("Starting training...")
        # Check if data needs feature extraction
        X_features = []
        for item in training_data:
            if item.shape == (1, 25088) or (len(item.shape)==1 and item.shape[0]==25088):
                 X_features.append(item.reshape(1, -1))
            else:
                 # Assume it's an image
                 feat = self.extract_features(item)
                 X_features.append(feat)
        
        X = np.vstack(X_features)
        y = np.array(labels)
        
        print(f"Fitting SVM with {len(y)} samples...")
        self.classifier.fit(X, y)
        print("Training complete.")
        
        # Save the model
        joblib.dump(self.classifier, self.classifier_path)
        print(f"Model saved to {self.classifier_path}")

# Example usage/helper for training script
if __name__ == "__main__":
    print("Initializing Analyzer...")
    analyzer = EmotionAnalyzer()
    print("Analyzer ready.")
