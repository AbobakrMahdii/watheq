import numpy as np
import cv2
from typing import Dict


class FaceService:
    def __init__(
        self,
        model_name: str = "Facenet",
        distance_metric: str = "cosine",
        threshold: float = 0.7,
    ):
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.threshold = threshold

    # ========================
    # Utils
    # ========================
    def _bytes_to_image(self, data: bytes):
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        return img

    def _extract_face(self, img):
        from deepface import DeepFace

        faces = DeepFace.extract_faces(
            img_path=img,
            target_size=(224, 224),
            detector_backend="retinaface",
            enforce_detection=False,
        )

        if not faces:
            raise RuntimeError("No face detected in image")

        return faces[0]["face"]

    # ========================
    # MAIN LOGIC (ID vs LIVE)
    # ========================
    def verify_id_vs_live(self, id_photo: bytes, live_photo: bytes) -> Dict:
        img_id = self._bytes_to_image(id_photo)
        img_live = self._bytes_to_image(live_photo)

        try:
            from deepface import DeepFace
        except Exception as e:
            raise RuntimeError(
                "DeepFace not ready. If using TF 2.20+, install: pip install tf-keras"
            ) from e

        # 🔹 قص الوجه من صورة البطاقة
        id_face = self._extract_face(img_id)

        try:
            result = DeepFace.verify(
                id_face,
                img_live,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                enforce_detection=False,
            )
        except Exception as e:
            raise RuntimeError("DeepFace.verify failed") from e

        distance = float(result.get("distance", 1.0))
        match = bool(result.get("verified", False))

        similarity = max(0.0, min(1.0, 1.0 - distance))

        return {
            "model": self.model_name,
            "distance_metric": self.distance_metric,
            "distance": distance,
            "threshold": self.threshold,
            "match": match,
            "similarity": similarity,
            "similarity_percent": similarity * 100.0,
        }