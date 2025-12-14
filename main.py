import cv2
import os
from emotion_analyzer import EmotionAnalyzer
import emotions

def main():
    # Initialize the analyzer
    analyzer = EmotionAnalyzer()
    
    # Path to sample image - Replace with your valid image path
    # If using your camera, you can adapt this to cv2.VideoCapture
    image_path = "sample_child.png"  
    
    if not os.path.exists(image_path):
        print(f"Sample image '{image_path}' not found.")
        print("Please place an image named 'sample_child.jpg' in this directory or update the path in 'main.py'.")
        # Create a dummy image for testing if it doesn't exist? 
        # Better to just ask user.
        return

    print(f"Analyzing {image_path}...")
    
    # Analyze
    original_img = cv2.imread(image_path)
    results = analyzer.detect_and_classify(original_img)

    if not results:
        print("No faces detected in the image.")
        return

    # If multiple faces present, process them all; otherwise process the first one
    # results is a list of dicts: {label, confidence, box, all_scores}
    for i, res in enumerate(results):
        label = res.get('label', 'Unknown')
        conf = res.get('confidence', 0.0)
        box = res.get('box', None)

        print(f"Result (subject {i+1}): {label} ({conf:.2%} confidence)")

        if isinstance(box, tuple):
            x, y, w, h = box
            # Normalize label to canonical form for consistent checks
            canonical = emotions.normalize_label(label) if isinstance(label, str) else None

            # Visual Output
            color = (0, 255, 0) # Green for neutral/positive
            if canonical in ["Sadness", "Fear", "Anger", "Disgust"]:
                color = (0, 0, 255) # Red for distress
                print("ALERT: Negative emotion detected! Notification sent to caretaker.")

            cv2.rectangle(original_img, (x, y), (x+w, y+h), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(original_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save output once after annotating all faces
    output_path = "processed_" + image_path
    cv2.imwrite(output_path, original_img)
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    main()
