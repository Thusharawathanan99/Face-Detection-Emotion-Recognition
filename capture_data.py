import cv2
import os
import time

def create_dataset_folders(base_path, categories):
    for category in categories:
        path = os.path.join(base_path, category)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")

def capture_data():
    base_path = "dataset"
    categories = ["happy", "sad", "angry", "neutral"]
    
    create_dataset_folders(base_path, categories)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("=== Face Emotion Data Capture Tool ===")
    print("Press the following keys to save an image for each emotion:")
    print("  'h' - Happy")
    print("  's' - Sad")
    print("  'a' - Angry")
    print("  'n' - Neutral")
    print("  'q' - Quit")
    
    counts = {cat: len(os.listdir(os.path.join(base_path, cat))) for cat in categories}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Display instructions on frame
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press: h(Happy), s(Sad), a(Angry), n(Neutral), q(Quit)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for i, cat in enumerate(categories):
            cv2.putText(display_frame, f"{cat}: {counts[cat]}", 
                        (10, 60 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Data Capture', display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        save_category = None
        if key == ord('h'):
            save_category = "happy"
        elif key == ord('s'):
            save_category = "sad"
        elif key == ord('a'):
            save_category = "angry"
        elif key == ord('n'):
            save_category = "neutral"
        elif key == ord('q'):
            break

        if save_category:
            timestamp = int(time.time() * 1000)
            filename = f"{save_category}_{timestamp}.jpg"
            save_path = os.path.join(base_path, save_category, filename)
            cv2.imwrite(save_path, frame)
            counts[save_category] += 1
            print(f"Saved {save_category} image: {filename}")
            
            # Visual feedback
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), 10)
            cv2.imshow('Data Capture', display_frame)
            cv2.waitKey(200) # Short pause for visual feedback

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_data()
