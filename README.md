# Face Detection and Emotion Recognition Module - Full System

## 🚀 Quick Start
To run the full demonstration immediately:

1.  **Install Requirements** (Wait for the background install to finish first):
    ```bash
    pip install -r requirements.txt
    ```
2.  **Initialize System**:
    Run this once to create a placeholder model file (so the app doesn't crash):
    ```bash
    python create_dummy_model.py
    ```
3.  **Start the Dashboard (Monitor)**:
    Open a new terminal and run:
    ```bash
    streamlit run caretaker_dashboard.py
    ```
4.  **Start the Camera (Sensor)**:
    Open *another* terminal and run:
    ```bash
    python camera_node.py
    ```

## 🏗 System Architecture
This "Full Project" consists of two networked components:
1.  **Camera Node (`camera_node.py`)**: Uses **OpenCV + MTCNN + VGG16** to process the video feed in real-time. It logs any "Distress" (Sad/Fear/Angry) events to a CSV file.
2.  **Caretaker Dashboard (`caretaker_dashboard.py`)**: A **Streamlit** web app that reads the CSV logs in real-time to display charts, stats, and alerts to the caretaker.

## 🛠 Real Training
The `create_dummy_model.py` script generates random predictions. To make it smart:
1.  Put your dataset in a `dataset/` folder.
2.  Run `python train_model.py`.
3.  Restart `camera_node.py`.

## 📂 File List
- `camera_node.py`: Real-time video processor.
- `caretaker_dashboard.py`: Web UI for alerts.
- `emotion_analyzer.py`: The core AI engine.
- `train_model.py`: Training script.
- `create_dummy_model.py`: Setup script for quick demo.

## 📚 Supported Emotions

This project ships a central list of emotion labels and short descriptions in `emotions.py`.
The following labels are included (human-readable descriptions available via the `get_description()` helper):

- Joy
- Sadness
- Anger
- Fear
- Love
- Disgust
- Surprise
- Shame
- Guilt
- Pride
- Envy
- Jealousy
- Grief
- Hope
- Loneliness
- Gratitude
- Anxiety
- Contentment
- Nostalgia
- Awe

If you're building or retraining models, ensure your dataset and classifier labels match these names exactly (case-insensitive matching is supported by the helper).

### Alias support

To remain backwards-compatible with common dataset folder names and older models, the project supports a set of aliases. Examples:

- `happy`, `joyful` -> `Joy`
- `angry` -> `Anger`
- `sad` -> `Sadness`
- `neutral` -> `Contentment` (mapped to the closest canonical label)

The helper `emotions.get_description(label)` and `EmotionAnalyzer.get_emotion_description(label)` will resolve these aliases case-insensitively and return the canonical human-readable description.
