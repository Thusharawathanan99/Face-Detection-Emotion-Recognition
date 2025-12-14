import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn import svm
import joblib
import os
import emotions

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
            # Record known classes (if the classifier exposes them)
            try:
                self.class_names = list(self.classifier.classes_)
            except Exception:
                self.class_names = []
        else:
            print("No pre-trained classifier found. You will need to train the model first.")
            # Initialize a new SVM for training later
            self.classifier = svm.SVC(kernel='linear', probability=True)
            self.class_names = []

    def preprocess_face(self, face_img):
        """
        Resizes and normalizes the face image for VGG16.
        """
        # Resize to VGG16 input size and convert to float32
        # Note: Keras' VGG16 `preprocess_input` expects pixel values in the
        # 0-255 range and internally converts from RGB->BGR and subtracts the
        # ImageNet mean. We therefore resize and cast to float32 and then
        # call preprocess_input.
        face_resized = cv2.resize(face_img, (224, 224)).astype('float32')
        # Ensure the image is in RGB order (face_img is expected to be RGB)
        face_preprocessed = preprocess_input(face_resized)
        # Add batch dimension (1, 224, 224, 3)
        face_batch = np.expand_dims(face_preprocessed, axis=0)
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
            return []

        results = []
        
        # Process ALL faces found
        for face in detections:
            try:
                x, y, width, height = face['box']
                x, y = max(0, x), max(0, y) # clamp coords
                
                # Crop the face
                face_img = rgb_img[y:y+height, x:x+width]
                
                if face_img.size == 0: continue

                # Stage 2: Feature Extraction & Classification
                if not hasattr(self.classifier, 'classes_'):
                     continue

                features = self.extract_features(face_img)

                # Get probabilities for all classes (if available)
                if hasattr(self.classifier, 'predict_proba'):
                    probs = self.classifier.predict_proba(features)[0]
                    classes = list(getattr(self.classifier, 'classes_', self.class_names))
                else:
                    # Fallback: classifier doesn't support predict_proba
                    # Try decision_function -> softmax
                    try:
                        scores = self.classifier.decision_function(features)[0]
                        exps = np.exp(scores - np.max(scores))
                        probs = exps / np.sum(exps)
                        classes = list(getattr(self.classifier, 'classes_', self.class_names))
                    except Exception:
                        # As a last resort, mark as unknown
                        probs = np.array([1.0])
                        classes = [getattr(self.classifier, 'classes_', ["Unknown"])[0]]

                # Map classifier class names to canonical emotion labels where possible
                emotion_probs = {}
                for i, cls in enumerate(classes):
                    try:
                        canon = emotions.normalize_label(str(cls))
                        key = canon if canon is not None else str(cls)
                    except Exception:
                        key = str(cls)
                    emotion_probs[key] = float(probs[i])

                # Find top emotion index
                best_idx = int(np.argmax(probs))
                raw_label = classes[best_idx]
                confidence = float(probs[best_idx])

                # Normalize the raw label to canonical project label when possible
                try:
                    canonical = emotions.normalize_label(raw_label)
                    emotion_label = canonical if canonical is not None else str(raw_label)
                except Exception:
                    emotion_label = str(raw_label)
                
                # Flag when the classifier only knows a single class — helpful
                # for UI warnings and preventing spurious logging.
                single_class = (len(classes) <= 1)

                results.append({
                    "label": emotion_label,
                    "confidence": confidence,
                    "box": (x, y, width, height),
                    "all_scores": emotion_probs,
                    "single_class": single_class
                })

            except Exception as e:
                print(f"Error processing a face: {e}")
                
        return results

    def get_emotion_description(self, label):
        """Return a human-readable description for a predicted emotion label.

        Uses the descriptions from `emotions.py`. This helper is non-invasive and
        returns a fallback string when the label is unknown.
        """
        try:
            return emotions.get_description(label)
        except Exception:
            return "No description available."

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
