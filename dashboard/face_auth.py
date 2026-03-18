import os
import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class FaceAuthLBPH:
    """
    Lightweight face authentication using:
      - Haar cascade for face detection
      - OpenCV LBPH for recognition

    Dataset layout (recommended):
      faces_db/
        Alice/
          1.jpg
          2.jpg
        Bob/
          1.jpg
    Also supports legacy flat layout:
      faces_db/Alice.jpg  (label = "Alice")
    """

    def __init__(
        self,
        faces_dir: str,
        model_path: str,
        labels_path: str,
        cascade_path: Optional[str] = None,
    ) -> None:
        self.faces_dir = faces_dir
        self.model_path = model_path
        self.labels_path = labels_path

        if cascade_path is None:
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Requires opencv-contrib-python
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError(
                "OpenCV face module not found. Install opencv-contrib-python and remove opencv-python."
            )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self.is_trained = False

        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.label_to_id = {k: int(v) for k, v in data.get("label_to_id", {}).items()}
            self.id_to_label = {int(k): v for k, v in data.get("id_to_label", {}).items()}

        if os.path.exists(self.model_path) and self.id_to_label:
            try:
                self.recognizer.read(self.model_path)
                self.is_trained = True
            except Exception:
                # Corrupt/old model file: force retrain
                self.is_trained = False

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.labels_path), exist_ok=True)
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump(
                {"label_to_id": self.label_to_id, "id_to_label": self.id_to_label},
                f,
                indent=2,
            )
        if self.is_trained:
            self.recognizer.write(self.model_path)

    def _iter_dataset_images(self) -> List[Tuple[str, str]]:
        """
        Returns list of (label, filepath) for all images in faces_dir.
        """
        items: List[Tuple[str, str]] = []
        if not os.path.isdir(self.faces_dir):
            return items

        # Folder-per-person
        for entry in os.listdir(self.faces_dir):
            full = os.path.join(self.faces_dir, entry)
            if os.path.isdir(full):
                label = entry
                for fn in os.listdir(full):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        items.append((label, os.path.join(full, fn)))

        # Flat files
        for fn in os.listdir(self.faces_dir):
            full = os.path.join(self.faces_dir, fn)
            if os.path.isfile(full) and fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                label = os.path.splitext(fn)[0]
                items.append((label, full))

        return items

    def _detect_faces_gray(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # (x, y, w, h)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        return list(faces) if faces is not None else []

    @staticmethod
    def _pick_largest_face(faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        if not faces:
            return None
        return max(faces, key=lambda b: b[2] * b[3])

    @staticmethod
    def _prep_face(gray: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        face = gray[y : y + h, x : x + w]
        face = cv2.equalizeHist(face)
        face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_LINEAR)
        return face

    def retrain(self) -> Dict[str, int]:
        """
        Rebuilds the LBPH model from faces_dir.
        Returns stats: {"labels": n, "images": n, "used_faces": n}
        """
        images: List[np.ndarray] = []
        labels: List[int] = []

        # Reset label maps
        self.label_to_id = {}
        self.id_to_label = {}

        used_faces = 0
        total_images = 0

        for label, path in self._iter_dataset_images():
            total_images += 1
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self._detect_faces_gray(gray)
            box = self._pick_largest_face(faces)
            if box is None:
                continue
            face = self._prep_face(gray, box)

            if label not in self.label_to_id:
                new_id = len(self.label_to_id) + 1  # start at 1
                self.label_to_id[label] = new_id
                self.id_to_label[new_id] = label
            lid = self.label_to_id[label]

            images.append(face)
            labels.append(lid)
            used_faces += 1

        if len(images) >= 2 and len(set(labels)) >= 1:
            self.recognizer.train(images, np.array(labels, dtype=np.int32))
            self.is_trained = True
        else:
            self.is_trained = False

        self._save()
        return {"labels": len(self.label_to_id), "images": total_images, "used_faces": used_faces}

    def enroll_image(self, name: str, bgr_img: np.ndarray) -> bool:
        """
        Stores an uploaded image under faces_dir/<name>/timestamp.jpg, returns True if saved.
        """
        safe = "".join([c for c in name.strip() if c.isalnum() or c in ("_", "-", " ")]).strip()
        if not safe:
            return False
        person_dir = os.path.join(self.faces_dir, safe)
        os.makedirs(person_dir, exist_ok=True)
        fn = f"{int(cv2.getTickCount())}.jpg"
        path = os.path.join(person_dir, fn)
        ok = cv2.imwrite(path, bgr_img)
        return bool(ok)

    def list_users(self) -> List[str]:
        if not os.path.isdir(self.faces_dir):
            return []
        users = []
        for entry in os.listdir(self.faces_dir):
            full = os.path.join(self.faces_dir, entry)
            if os.path.isdir(full):
                users.append(entry)
        # Also include flat-file labels
        for label, path in self._iter_dataset_images():
            if os.path.dirname(path) == self.faces_dir and label not in users:
                users.append(label)
        return sorted(set(users))

    def delete_user(self, name: str) -> bool:
        """
        Deletes faces_dir/<name>/ if it exists.
        (Does not delete legacy flat files.)
        """
        person_dir = os.path.join(self.faces_dir, name)
        if not os.path.isdir(person_dir):
            return False
        for root, _, files in os.walk(person_dir):
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass
        try:
            os.rmdir(person_dir)
        except OSError:
            # Non-empty or locked; best-effort
            pass
        return True

    def identify_from_bgr(
        self,
        bgr_img: np.ndarray,
        *,
        threshold: float = 65.0,
    ) -> Tuple[Optional[str], Optional[float], bool]:
        """
        Strict identification:
          - If no face detected: returns (None, None, False)
          - If model not trained: returns (None, None, True)  (face visible but no model)
          - Else: returns (label or None, confidence, True)

        Note: For LBPH, *lower* confidence values mean better matches.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces_gray(gray)
        box = self._pick_largest_face(faces)
        if box is None:
            return None, None, False

        if not self.is_trained:
            return None, None, True

        face = self._prep_face(gray, box)
        pred_id, conf = self.recognizer.predict(face)
        label = self.id_to_label.get(int(pred_id))
        if label is None:
            return None, float(conf), True

        if float(conf) <= float(threshold):
            return label, float(conf), True
        return None, float(conf), True


