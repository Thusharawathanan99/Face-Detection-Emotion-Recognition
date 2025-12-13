from emotion_analyzer import EmotionAnalyzer
import os
import cv2
import numpy as np

def load_dataset(dataset_path):
    """
    Loads images and labels from a directory structure.
    Expected structure:
    dataset/
        happy/
            img1.jpg
            ...
        sad/
            img1.jpg
            ...
        ...
    """
    images = []
    labels = []
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return [], []

    classes = os.listdir(dataset_path)
    print(f"Found classes: {classes}")

    for label in classes:
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Loading {label}...")
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # We can store the raw image and let the analyzer preprocess,
                    # or preprocess here to save memory if dataset is huge.
                    # For PoC, let's just store the image.
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    return images, labels

def main():
    dataset_path = "dataset" # Update this to your LIRIS-CSE dataset path
    
    if not os.path.exists(dataset_path):
        print("Dataset directory not found.")
        print(f"Please create a folder named '{dataset_path}' with subfolders for each emotion (e.g., 'happy', 'sad').")
        return

    print("Loading dataset...")
    X, y = load_dataset(dataset_path)
    
    if len(X) == 0:
        print("No images found.")
        return

    analyzer = EmotionAnalyzer()
    
    # In a real scenario, you heavily augment data here.
    
    print("Training model...")
    analyzer.train(X, y)
    print("Training finished.")

if __name__ == "__main__":
    main()
