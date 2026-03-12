from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8m.pt")  

    def detect(self, frame):
        return self.model.track(
            frame,
            stream=True,
            persist=True,     
            conf=0.5,
            iou=0.6,
            device=0
        )
