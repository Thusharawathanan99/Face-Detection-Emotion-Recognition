import numpy as np
from sklearn import svm
import joblib
from tensorflow.keras.applications.vgg16 import VGG16
import os

print("Creating Dummy Model for Demonstration...")

# 1. Generate Fake "Features" (mocking VGG16 output size)
# VGG16 flattened vector is typically around 25088 for 224x224 input -> 7x7x512
feature_size = 25088 
num_samples = 50

X_dummy = np.random.rand(num_samples, feature_size)
y_dummy = np.random.choice(['happy', 'sad', 'neutral', 'angry'], num_samples)

# 2. Train a fast SVM
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_dummy, y_dummy)

# 3. Save
joblib.dump(clf, 'emotion_classifier.pkl')
print("Dummy 'emotion_classifier.pkl' created.")
print("WARNING: This model detects RANDOM emotions. Use 'train_model.py' with real data for actual results.")
