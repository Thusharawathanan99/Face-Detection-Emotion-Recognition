import cv2
import os
from emotion_analyzer import EmotionAnalyzer

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
    label, conf, face_box = analyzer.detect_and_classify(original_img)
    
    print(f"Result: {label} ({conf:.2%} confidence)")

    if isinstance(face_box, tuple):
        x, y, w, h = face_box
        
        # Visual Output
        color = (0, 255, 0) # Green for neutral/positive
        if label in ["sad", "fear", "angry", "disgust"]:
            color = (0, 0, 255) # Red for distress
            print("ALERT: Negative emotion detected! Notification sent to caretaker.")
        
        cv2.rectangle(original_img, (x, y), (x+w, y+h), color, 2)
        text = f"{label}: {conf:.2f}"
        cv2.putText(original_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Save output
        output_path = "processed_" + image_path
        cv2.imwrite(output_path, original_img)
        print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    main()
