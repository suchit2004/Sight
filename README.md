SafeSight: AI-Based Smart Surveillance System

SafeSight is a real-time surveillance system built using computer vision and AI to detect critical events such as intrusion, crowd formation, fire, and human falls, and trigger alerts instantly.

Features
Person Detection & Tracking using YOLO
Intrusion Detection with custom ROI
Loitering Detection based on dwell time
Fire Detection using color-based segmentation
Crowd Detection using grid density and distance methods
Fall Detection using MediaPipe Pose
Face Recognition (Optional) using LBPH
Snapshots & Event Logging with CSV export
Telegram Alerts with image notifications
Live Metrics (FPS, latency, alerts)

Tech Stack
Python
OpenCV
YOLO (Ultralytics)
MediaPipe
Flask
NumPy


Project Structure

SafeSight/
├── app/
│   ├── main.py
│   ├── templates/
│   └── static/snapshots/
├── detection/
│   └── yolo_detector.py
├── data/
│   ├── videos/
│   └── faces/
├── requirements.txt
└── README.md


Installation
git clone https://github.com/your-username/safesight.git
cd safesight
pip install -r requirements.txt
python main.py

To Run 
cd dashboard
python app.py

Open in browser:
http://127.0.0.1:5000/

Key Endpoints
/video_feed – Live stream
/toggle_crowd – Enable/disable crowd detection
/toggle_fire – Enable/disable fire detection
/toggle_fall – Enable/disable fall detection
/toggle_face – Enable/disable face recognition
/events – View logs
/metrics – Live system metrics
Telegram Setup

Add your bot credentials in the code:

BOT_TOKEN = "YOUR_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
