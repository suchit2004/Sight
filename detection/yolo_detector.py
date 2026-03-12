from ultralytics import YOLO
import numpy as np


class YOLODetector:
    def __init__(self):
        # Phase 2 — YOLOv8m-seg replaces YOLOv8m
        # Same architecture, same CUDA pipeline, same ultralytics API
        # Adds per-instance pixel masks with zero extra dependencies
        # Model auto-downloads on first run (~52MB)
        self.model = YOLO("yolov8m-seg.pt")

    def detect(self, frame):
        """
        Returns ultralytics Results — identical API to yolov8m.pt
        Each result now also contains result.masks (None if no masks found)
        Everything in app.py that uses result.boxes is unchanged.
        """
        return self.model.track(
            frame,
            stream=True,
            persist=True,
            conf=0.5,
            iou=0.6,
            device=0
        )

    @staticmethod
    def extract_masks(result, frame_shape):
        """
        Extract per-track pixel masks from a segmentation result.

        Returns dict: { track_id → binary mask (H x W, dtype=uint8) }

        Binary mask values:
            255 = pixel belongs to this object
            0   = pixel does not belong to this object

        Returns empty dict if:
            - result.masks is None (no segmentation available)
            - result.boxes has no valid track IDs
        """
        masks_by_id = {}

        if result.masks is None:
            return masks_by_id

        h, w = frame_shape[:2]

        for i, box in enumerate(result.boxes):
            if box.id is None:
                continue

            track_id = int(box.id.item())

            # result.masks.data is shape (N, mask_h, mask_w) — float32 0-1
            # Index i corresponds to box i
            if i >= len(result.masks.data):
                continue

            # Resize mask to full frame resolution and binarise
            mask_tensor = result.masks.data[i].cpu().numpy()  # float32
            mask_resized = (mask_tensor * 255).astype(np.uint8)

            import cv2
            if mask_resized.shape != (h, w):
                mask_resized = cv2.resize(
                    mask_resized, (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            # Binarise: any value > 127 = object pixel
            _, binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            masks_by_id[track_id] = binary

        return masks_by_id