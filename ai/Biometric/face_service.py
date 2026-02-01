
import numpy as np
import cv2
from typing import Dict
#athhh

class FaceService:
    def __init__(self, model_name: str = "Facenet", distance_metric: str = "cosine", threshold: float = 0.7):
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.threshold = threshold

    def _bytes_to_image(self, data: bytes):
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        return img

    def verify_faces(self, photo1: bytes, photo2: bytes) -> Dict:
        img1 = self._bytes_to_image(photo1)
        img2 = self._bytes_to_image(photo2)

        try:
            from deepface import DeepFace  # lazy import (can fail if deps/models missing)
        except Exception as e:
            raise RuntimeError(
                "DeepFace is not ready on the server. "
                "If you are using TensorFlow 2.20+, install `tf-keras` (pip install tf-keras)."
            ) from e

        try:
            result = DeepFace.verify(
                img1,
                img2,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                enforce_detection=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"DeepFace.verify failed ({type(e).__name__}). "
                "This can happen if no face is detectable in the document photo, "
                "or if DeepFace dependencies/models are not ready on the server."
            ) from e

        if not isinstance(result, dict):
            raise RuntimeError("DeepFace.verify returned unexpected result type")

        try:
            distance = float(result.get("distance", 1.0))
            match = bool(result.get("verified", False))
        except Exception as e:
            raise RuntimeError("Invalid DeepFace result payload") from e
        similarity = 1.0 - distance
        similarity = float(max(0.0, min(1.0, similarity)))
        similarity_percent = similarity * 100.0

        return {
            "model": self.model_name,
            "distance_metric": self.distance_metric,
            "distance": distance,
            "threshold": self.threshold,
            "match": match,
            "similarity": similarity,
            "similarity_percent": similarity_percent,
        }
