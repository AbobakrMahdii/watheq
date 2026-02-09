import numpy as np
import cv2
from typing import Dict


class FaceService:
    def __init__(
        self,
        model_name: str = "Facenet",
        distance_metric: str = "cosine",
        accept_threshold_percent: float = 80.0,  # ✅ نسبة القبول
        id_score_threshold: float = 0.11,  # ✅ عتبة تصنيف البطاقة
    ):
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.accept_threshold_percent = accept_threshold_percent
        self.id_score_threshold = id_score_threshold

    # ---------------- Utils ----------------
    def _bytes_to_image(self, data: bytes):
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        return img

    # -------- ID / LIVE heuristic ----------
    def _id_likeness_score(self, img) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.mean(edges > 0))

        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        textish = float(np.mean(morph > 0))

        score = 0.55 * edge_density + 0.45 * textish
        return max(0.0, min(1.0, score))

    def _is_id_card(self, img) -> bool:
        return self._id_likeness_score(img) >= self.id_score_threshold

    # -------- Face extraction (standalone use) ----
    def _extract_face(self, img):
        """Extract and return a face crop as uint8 [0,255] BGR image.

        NOTE: DeepFace.extract_faces returns float [0,1] normalised pixels.
        We convert back to uint8 so the result is safe to pass into any
        downstream function that expects a standard OpenCV image.
        """
        from deepface import DeepFace

        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=False,
        )

        if not faces:
            raise RuntimeError("No face detected in ID image")

        face = faces[0]["face"]  # float [0,1]
        face = (face * 255).astype(np.uint8)  # → uint8 [0,255]
        face = cv2.resize(face, (224, 224))
        return face

    # ------------- MAIN -------------------
    def verify_id_vs_live(self, photo1: bytes, photo2: bytes) -> Dict:
        img1 = self._bytes_to_image(photo1)
        img2 = self._bytes_to_image(photo2)

        score1 = self._id_likeness_score(img1)
        score2 = self._id_likeness_score(img2)

        is_id_1 = score1 >= self.id_score_threshold
        is_id_2 = score2 >= self.id_score_threshold

        # ❌ لايف + لايف
        if not is_id_1 and not is_id_2:
            raise ValueError(
                f"Rejected: both inputs look like LIVE photos "
                f"(scores: {score1:.3f}, {score2:.3f}). Required: ID + LIVE."
            )

        # ❌ بطاقة + بطاقة
        if is_id_1 and is_id_2:
            raise ValueError(
                f"Rejected: both inputs look like ID cards "
                f"(scores: {score1:.3f}, {score2:.3f}). Required: ID + LIVE."
            )

        # ✅ تحديد البطاقة واللايف
        if is_id_1:
            img_id, img_live = img1, img2
            id_score, live_score = score1, score2
        else:
            img_id, img_live = img2, img1
            id_score, live_score = score2, score1

        from deepface import DeepFace

        # Let DeepFace handle face detection + alignment + embedding for
        # BOTH images internally with RetinaFace.  This avoids the old bug
        # where _extract_face returned float [0,1] pixels that DeepFace.verify
        # then mis-preprocessed (expecting uint8 [0,255]), producing garbage
        # embeddings and <10% similarity scores.
        result = DeepFace.verify(
            img_id,
            img_live,
            model_name=self.model_name,
            distance_metric=self.distance_metric,
            detector_backend="retinaface",
            enforce_detection=False,
        )

        distance = float(result.get("distance", 1.0))
        similarity = max(0.0, min(1.0, 1.0 - distance))
        similarity_percent = round(similarity * 100, 2)

        accepted = similarity_percent >= self.accept_threshold_percent

        return {
            "similarity_percent": similarity_percent,
            "accepted": accepted,
            "accept_threshold_percent": self.accept_threshold_percent,
            "debug": {
                "id_likeness_score": round(id_score, 4),
                "live_likeness_score": round(live_score, 4),
            },
        }
