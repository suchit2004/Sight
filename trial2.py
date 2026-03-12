import cv2
import numpy as np
from detection.yolo_detector import YOLODetector


detector = YOLODetector()

cap = cv2.VideoCapture("../data/CCTV1.mp4")

roi_points = [(100, 10), (500, 10), (500, 700), (100, 700)]
roi_polygon = np.array(roi_points, np.int32)


while True:
    ret, frame = cap.read()

    if not ret:
        print("End of CCTV footage")
        break

    results_gen = detector.detect(frame)
    results = list(results_gen)  

    annotated_frame = results[0].plot()

    cv2.polylines(annotated_frame, [roi_polygon], True, (255, 0, 0), 2)

    for box in results[0].boxes:
        class_id = int(box.cls[0])

        if class_id == 0:  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False)

            if inside >= 0:
                cv2.putText(
                    annotated_frame,
                    "INTRUSION DETECTED",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("SafeSight - CCTV Analysis", annotated_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
