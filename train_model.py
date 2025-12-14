from emotion_analyzer import EmotionAnalyzer
import os
import cv2
import numpy as np
import joblib

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

    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Found classes: {classes}")

    for label in classes:
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Loading {label}...")
        # Gather files then iterate (allows optional sampling later)
        file_list = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        for img_name in file_list:
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    return images, labels

def main(dataset_path="dataset", max_per_class=2000, random_seed=42):
    
    if not os.path.exists(dataset_path):
        print("Dataset directory not found.")
        print(f"Please create a folder named '{dataset_path}' with subfolders for each emotion (e.g., 'happy', 'sad').")
        return

    print("Loading dataset...")
    X_all, y_all = load_dataset(dataset_path)
    if len(X_all) == 0:
        print("No images found in dataset.")
        return

    # Optionally subsample per-class to avoid memory explosions on large datasets.
    print(f"Subsampling up to {max_per_class} images per class (if available)...")
    from collections import defaultdict
    per_class_imgs = defaultdict(list)
    for img, lbl in zip(X_all, y_all):
        per_class_imgs[lbl].append(img)

    X, y = [], []
    import random
    random.seed(random_seed)
    for lbl, imgs in per_class_imgs.items():
        if max_per_class is not None and len(imgs) > max_per_class:
            chosen = random.sample(imgs, max_per_class)
        else:
            chosen = imgs
        X.extend(chosen)
        y.extend([lbl] * len(chosen))

    analyzer = EmotionAnalyzer()
    
    # In a real scenario, you heavily augment data here.
    
    print("Training model...")
    # Split data to evaluate performance
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    
    # We need to extract features FIRST to split properly if we want to validata on features, 
    # but analyzer.train handles extraction.
    # To get best output, let's extract features manually here using analyzer's helper, then split.
    
    print("Extracting features for all images (this may take a while)...")
    X_features = []
    valid_labels = []
    
    for i, img in enumerate(X):
        try:
            # Convert BGR (cv2.imread) -> RGB expected by analyzer
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feat = analyzer.extract_features(img_rgb)
            X_features.append(feat)
            valid_labels.append(y[i])
        except Exception as e:
            print(f"Skipping bad image index {i}: {e}")
            
    if not X_features:
        print("No valid features extracted.")
        return

    X_data = np.vstack(X_features)
    y_data = np.array(valid_labels)

    # Stratified split to ensure all classes are represented
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data, random_state=42)
    except ValueError:
        # Fallback if some class has too few samples (e.g. only 1 image)
        print("Warning: Not enough data per class for stratified split. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    print(f"Training on {len(X_train)} samples, Validating on {len(X_test)} samples...")
    analyzer.classifier.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = analyzer.classifier.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Retrain on FULL dataset for final production model
    print("\nRetraining on full dataset for maximum performance...")
    analyzer.classifier.fit(X_data, y_data)
    joblib.dump(analyzer.classifier, analyzer.classifier_path)
    print(f"Final model saved to {analyzer.classifier_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset")
    parser.add_argument("--max-per-class", type=int, default=2000, help="Max images to use per class (to limit memory)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    main(dataset_path=args.dataset, max_per_class=args.max_per_class, random_seed=args.seed)
