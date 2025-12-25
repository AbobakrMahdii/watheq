
import numpy as np
import cv2
from typing import Dict
from deepface import DeepFace


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

        result = DeepFace.verify(
            img1,
            img2,
            model_name=self.model_name,
            distance_metric=self.distance_metric,
            enforce_detection=False,
        )

        distance = float(result["distance"])
        match = bool(result["verified"])
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