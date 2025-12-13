import cv2
import time
import pandas as pd
from datetime import datetime
from emotion_analyzer import EmotionAnalyzer
import os

class CameraNode:
    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.analyzer = EmotionAnalyzer()
        self.is_running = False
        self.alert_log_file = "alert_log.csv"
        self.initialize_log()

    def initialize_log(self):
        if not os.path.exists(self.alert_log_file):
            df = pd.DataFrame(columns=["timestamp", "emotion", "confidence", "status"])
            df.to_csv(self.alert_log_file, index=False)

    def log_alert(self, emotion, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "ALERT" if emotion in ["sad", "fear", "angry", "disgust"] else "NORMAL"
        
        new_entry = pd.DataFrame([{
            "timestamp": timestamp,
            "emotion": emotion,
            "confidence": confidence,
            "status": status
        }])
        
        # Append to CSV (efficient enough for PoC)
        new_entry.to_csv(self.alert_log_file, mode='a', header=False, index=False)

    def start(self):
        cap = cv2.VideoCapture(self.camera_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        print("Starting Camera Node... Press 'q' to quit.")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Optimization: Analyze every 5th frame to reduce lag
            frame_count += 1
            if frame_count % 5 != 0:
                cv2.imshow('Smart Caretaker Camera Node', frame) # Show raw frame in between
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Analyze frame
            label, conf, face_box = self.analyzer.detect_and_classify(frame)

            # Visualization
            display_frame = frame.copy()
            if isinstance(face_box, tuple):
                x, y, w, h = face_box
                color = (0, 255, 0)
                if label in ["sad", "fear", "angry"]:
                    color = (0, 0, 255)
                    # Log negative emotions
                    self.log_alert(label, conf)
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, f"{label}: {conf:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow('Smart Caretaker Camera Node', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = CameraNode()
    node.start()
