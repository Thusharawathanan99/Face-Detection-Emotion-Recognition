import cv2
import time
import pandas as pd
from datetime import datetime
from emotion_analyzer import EmotionAnalyzer
import os
import threading
try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False
    print("Warning: winsound not found. Audio alerts will be disabled.")

class CameraNode:
    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.analyzer = EmotionAnalyzer()
        self.is_running = False
        self.alert_log_file = "alert_log.csv"
        self.initialize_log()
        
        # DEMO MODE: Real output enabled
        self.simulation_mode = False 
        # Full list of supported emotions
        self.all_emotions = [
            "Joy", "Sadness", "Anger", "Fear", "Love", "Disgust", "Surprise", 
            "Shame", "Guilt", "Pride", "Envy", "Jealousy", "Grief", "Hope", 
            "Loneliness", "Gratitude", "Anxiety", "Contentment", "Nostalgia", "Awe"
        ]

    def initialize_log(self):
        if not os.path.exists(self.alert_log_file):
            df = pd.DataFrame(columns=["timestamp", "emotion", "confidence", "status"])
            df.to_csv(self.alert_log_file, index=False)

    def log_alert(self, emotion, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Extended list of negative emotions for alerts (Simplified)
        negative_emotions = [
            "Sad", "Angry", "Scared", "Yuck", "Ashamed", "Sorry", 
            "Jealous", "Crying", "Lonely", "Worried"
        ]
        status = "ALERT" if emotion in negative_emotions else "NORMAL"
        
        new_entry = pd.DataFrame([{
            "timestamp": timestamp,
            "emotion": emotion,
            "confidence": confidence,
            "status": status
        }])
        
        # Append to CSV (efficient enough for PoC)
        new_entry.to_csv(self.alert_log_file, mode='a', header=False, index=False)

    def start(self):
        # Import PIL here safely
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            has_pil = True
            # Windows Emoji Font Path
            font_path = "C:\\Windows\\Fonts\\seguiemj.ttf"
            try:
                font = ImageFont.truetype(font_path, 40)
            except:
                print("Emoji font not found, falling back to default.")
                font = ImageFont.load_default()
        except ImportError:
            has_pil = False
            print("Pillow not installed. Emojis will not be shown.")

        # Simple "Easy English" Map
        self.simple_map = {
            # Standard mappings
            "Joy": "Happy", "Sadness": "Sad", "Anger": "Angry", "Fear": "Scared",
            "Love": "Love", "Disgust": "Yuck", "Surprise": "Wow", "Shame": "Ashamed",
            "Guilt": "Sorry", "Pride": "Proud", "Envy": "Jealous", "Jealousy": "Jealous",
            "Grief": "Crying", "Hope": "Hopeful", "Loneliness": "Lonely", 
            "Gratitude": "Thankful", "Anxiety": "Worried", "Contentment": "Happy",
            "Nostalgia": "Missing", "Awe": "Wow", "Neutral": "Neutral",
            # Lowercase mappings for new dataset
            "happy": "Happy", "sad": "Sad", "angry": "Angry", "neutral": "Neutral"
        }

        emotion_emojis = {
            "Happy": "😊", "Sad": "😢", "Angry": "😠", "Scared": "😨",
            "Love": "❤️", "Yuck": "🤢", "Wow": "😲", "Ashamed": "😳",
            "Sorry": "😓", "Proud": "😌", "Jealous": "😒", "Crying": "😭",
            "Hopeful": "🤞", "Lonely": "🙍", "Thankful": "🙏", "Worried": "😰",
            "Missing": "🌇"
        }

        # Try to initialize camera with robust fallback
        cap = None
        potential_indexes = [self.camera_source]
        if self.camera_source == 0:
            potential_indexes.append(1) # Try secondary camera if default fails
            
        for source in potential_indexes:
            print(f"Attempting to open camera {source}...")
            # Try DirectShow (often better for Windows)
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"Successfully connected to camera {source} (DirectShow)")
                break
            cap.release()
            
            # Try Default backend
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                print(f"Successfully connected to camera {source} (Default)")
                break
            cap.release()
        
        if cap is None or not cap.isOpened():
            print("CRITICAL ERROR: Could not open video source.")
            print("Troubleshooting:")
            print("1. Ensure no other application (Zoom, Teams, other terminals) is using the camera.")
            print("2. Check if the camera is plugged in.")
            return

        print("Starting Camera Node with Cinematic Interface... Press 'q' to quit.")
        
        # Cinematic Variables
        scan_line_y = 0
        scan_direction = 1
        frame_count = 0
        
        # Try to load a "tech" font for the HUD
        hud_font = None
        sticker_font = None
        try:
             # Consolas for text
             hud_font = ImageFont.truetype("consola.ttf", 20)
             title_font = ImageFont.truetype("consola.ttf", 30)
             # Segoe UI Emoji for the sticker (Standard Windows Emoji font)
             sticker_font = ImageFont.truetype("seguiemj.ttf", 80) 
        except:
             try:
                hud_font = ImageFont.truetype("arial.ttf", 20)
                title_font = ImageFont.truetype("arial.ttf", 30)
                sticker_font = ImageFont.truetype("arial.ttf", 80)
             except:
                hud_font = ImageFont.load_default()
                title_font = ImageFont.load_default()
                sticker_font = ImageFont.load_default() # Fallback likely won't be large

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Create a clean canvas for the cinematic output
            display_frame = frame.copy()
            h_img, w_img = display_frame.shape[:2]

            # Optimization: Analyze every 10th frame for stability
            frame_count += 1
            if frame_count % 10 == 0:
                # Analyze frame - now returns a list of dicts
                self.last_results = self.analyzer.detect_and_classify(frame)
                
                # Check for Alert in ANY face
                any_alert = False
                negative_emotions = [
                    "Sad", "Angry", "Scared", "Yuck", "Ashamed", "Sorry", 
                    "Jealous", "Crying", "Lonely", "Worried"
                ]

                # Update status for each face
                for res in self.last_results:
                    raw_lbl = res['label']
                    simple_lbl = self.simple_map.get(raw_lbl, raw_lbl)
                    res['simple_label'] = simple_lbl # Store for drawing
                    
                    if simple_lbl in negative_emotions:
                        any_alert = True
                        res['is_alert'] = True
                        # Only actually log alerts when the classifier appears trained
                        # (i.e., has more than one class). This prevents spamming the
                        # alert log when a broken/untrained classifier always predicts
                        # the same class.
                        if not res.get('single_class', False):
                            self.log_alert(simple_lbl, res['confidence'])
                        else:
                            # mark but do not persist noisy alerts
                            res['logged_as_warning'] = True
                    else:
                        res['is_alert'] = False

                if any_alert and HAS_SOUND:
                     try:
                         # Two-tone alert
                         def play_alert():
                             winsound.Beep(600, 150)
                             winsound.Beep(450, 250)
                         # Prevent sound overlap/spam
                         if threading.active_count() < 5: 
                            threading.Thread(target=play_alert, daemon=True).start()
                     except: pass

            # Retrieve processed results
            results = getattr(self, 'last_results', [])
            
            # --- CINEMATIC DRAWING ---
            
            # Letterbox
            bar_height = int(h_img * 0.1)
            cv2.rectangle(display_frame, (0, 0), (w_img, bar_height), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (0, h_img - bar_height), (w_img, h_img), (0, 0, 0), -1)

            # System Status Text
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(display_frame, f"REC | {timestamp}", (20, bar_height - 15), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
            
            cv2.putText(display_frame, f"SYSTEM: ONLINE | TARGETS: {len(results)}", (w_img - 250, bar_height - 15), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            # Draw OpenCV primitives first (Lines/Boxes) for all faces
            for i, res in enumerate(results):
                x, y, w, h = res['box']
                is_alert = res.get('is_alert', False)
                color = (0, 0, 255) if is_alert else (255, 255, 0)
                
                # Corner Brackets
                line_len = int(w * 0.2)
                pts = [
                    ((x, y), (x + line_len, y)), ((x, y), (x, y + line_len)),
                    ((x + w, y), (x + w - line_len, y)), ((x + w, y), (x + w, y + line_len)),
                    ((x, y + h), (x + line_len, y + h)), ((x, y + h), (x, y + h - line_len)),
                    ((x + w, y + h), (x + w - line_len, y + h)), ((x + w, y + h), (x + w, y + h - line_len))
                ]
                for p1, p2 in pts:
                    cv2.line(display_frame, p1, p2, color, 2)
                
                # Scanline
                scan_line_y += (5 * scan_direction)
                real_scan_y = y + (frame_count % h)
                if y <= real_scan_y <= y + h:
                    cv2.line(display_frame, (x, real_scan_y), (x + w, real_scan_y), color, 1)

                # Panel Background
                panel_w, panel_h = 300, 110
                panel_x = x + w + 10
                panel_y = y
                if panel_x + panel_w > w_img: panel_x = x - (panel_w + 10)
                
                sub = display_frame.copy()
                cv2.rectangle(sub, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
                cv2.addWeighted(sub, 0.6, display_frame, 0.4, 0, display_frame)

            # Draw PIL Text for all faces
            if has_pil:
                img_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                # If classifier only has a single class, show a prominent warning
                # so the operator knows the model needs retraining.
                try:
                    classifier_warning = not self.analyzer.class_names or len(self.analyzer.class_names) <= 1
                except Exception:
                    classifier_warning = True
                if classifier_warning:
                    w_warn, h_warn = w_img - 40, 40
                    draw.rectangle([(20, bar_height + 10), (20 + w_warn, bar_height + 10 + h_warn)], fill=(40, 10, 10))
                    draw.text((30, bar_height + 15), "WARNING: Classifier may be undertrained (single class). Retrain for reliable results.", font=hud_font, fill=(255, 200, 200))
                
                for i, res in enumerate(results):
                    x, y, w, h = res['box']
                    label = res.get('simple_label', 'Unknown')
                    conf = res.get('confidence', 0.0)
                    is_alert = res.get('is_alert', False)
                    color = (0, 0, 255) if is_alert else (255, 255, 0)
                    pil_color = (color[2], color[1], color[0])
                    white = (255, 255, 255)
                    
                    panel_w = 300
                    panel_x = x + w + 10
                    if panel_x + panel_w > w_img: panel_x = x - (panel_w + 10)
                    panel_y = y
                    
                    draw.text((panel_x + 10, panel_y + 10), f"SUBJECT {i+1}", font=hud_font, fill=white)
                    draw.text((panel_x + 10, panel_y + 35), f"{label.upper()}", font=title_font, fill=pil_color)
                    
                    # Bar
                    bar_w = 160
                    draw.rectangle([panel_x + 10, panel_y + 80, panel_x + 10 + bar_w, panel_y + 80 + 6], outline=white, width=1)
                    draw.rectangle([panel_x + 10, panel_y + 80, panel_x + 10 + int(bar_w*conf), panel_y + 80 + 6], fill=pil_color)
                    
                    # Emoji
                    emoji = emotion_emojis.get(label, "")
                    draw.text((panel_x + 190, panel_y + 10), emoji, font=sticker_font, fill=white)
                
                display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow('Smart Caretaker - Cinematic Vision', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = CameraNode()
    node.start()
