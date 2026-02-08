"""
Font & Text Style Analyzer for Document Verification.

محلل الخطوط والأنماط النصية للتحقق من الوثائق:
- يتعلم الخصائص البصرية للنصوص من الصور المرجعية
- يقيس سمك الحروف، ارتفاعها، التباعد، الكثافة، والحدة
- يقارن النص في الوثيقة المقدمة بما تعلمه

Learns font profiles from reference document text regions and verifies
that submitted documents match the learned visual characteristics.

Usage:
    # Learn from reference
    analyzer = FontAnalyzer()
    profile = analyzer.learn_font_profile(reference_crop)
    analyzer.save_profile(profile, "fonts/identity_text_name.json")

    # Verify
    profile = FontAnalyzer.load_profile("fonts/identity_text_name.json")
    score, details = analyzer.verify_font(input_crop, profile)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FontProfile:
    """Learned visual characteristics of text in a document region."""

    # Stroke width statistics (from distance transform)
    stroke_width_mean: float = 0.0
    stroke_width_std: float = 0.0
    stroke_width_min: float = 0.0
    stroke_width_max: float = 0.0

    # Character height distribution
    char_height_mean: float = 0.0
    char_height_std: float = 0.0

    # Spacing
    inter_char_spacing_mean: float = 0.0
    inter_char_spacing_std: float = 0.0
    line_spacing_mean: float = 0.0

    # Density & sharpness
    ink_density: float = 0.0  # ratio of ink pixels to total
    edge_sharpness: float = 0.0  # Laplacian variance
    text_uniformity: float = 0.0  # how consistent the ink distribution is

    # Histogram features (normalized grayscale histogram)
    gray_histogram: list = field(default_factory=lambda: [0.0] * 16)

    # Metadata
    element_class: str = ""
    doc_type: str = ""
    image_size: tuple = (0, 0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FontProfile":
        d = d.copy()
        if "gray_histogram" in d and isinstance(d["gray_histogram"], list):
            d["gray_histogram"] = d["gray_histogram"]
        if "image_size" in d and isinstance(d["image_size"], list):
            d["image_size"] = tuple(d["image_size"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FontAnalyzer:
    """Analyze and verify text font/style properties in document images."""

    @staticmethod
    def _to_binary(gray: np.ndarray) -> np.ndarray:
        """Convert grayscale to binary (ink=255, background=0)."""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _compute_stroke_width(binary: np.ndarray) -> Dict[str, float]:
        """Compute stroke width stats using distance transform."""
        if binary.sum() == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        # Only consider ink pixels
        strokes = dist[binary > 0]
        if len(strokes) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        return {
            "mean": float(np.mean(strokes)),
            "std": float(np.std(strokes)),
            "min": float(np.min(strokes)),
            "max": float(np.max(strokes)),
        }

    @staticmethod
    def _compute_char_heights(binary: np.ndarray) -> Dict[str, float]:
        """Estimate character heights from connected components."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        heights = []
        min_area = binary.shape[0] * binary.shape[1] * 0.001  # min 0.1% of image

        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            if area > min_area and 0.2 < h / (w + 1e-6) < 5.0:
                heights.append(float(h))

        if not heights:
            return {"mean": 0, "std": 0}

        return {"mean": float(np.mean(heights)), "std": float(np.std(heights))}

    @staticmethod
    def _compute_spacing(binary: np.ndarray) -> Dict[str, float]:
        """Estimate inter-character and line spacing from projection profiles."""
        h, w = binary.shape

        # Vertical projection (for inter-character spacing)
        v_proj = binary.sum(axis=0) / 255.0
        # Find gaps (columns with near-zero ink)
        threshold = v_proj.max() * 0.05
        is_gap = v_proj < threshold
        gaps = []
        in_gap = False
        gap_start = 0
        for x in range(w):
            if is_gap[x] and not in_gap:
                in_gap = True
                gap_start = x
            elif not is_gap[x] and in_gap:
                in_gap = False
                gap_len = x - gap_start
                if 1 < gap_len < w * 0.3:  # reasonable gap
                    gaps.append(float(gap_len))

        inter_char = float(np.mean(gaps)) if gaps else 0.0
        inter_char_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0

        # Horizontal projection (for line spacing)
        h_proj = binary.sum(axis=1) / 255.0
        threshold = h_proj.max() * 0.05
        is_gap = h_proj < threshold
        line_gaps = []
        in_gap = False
        for y in range(h):
            if is_gap[y] and not in_gap:
                in_gap = True
                gap_start = y
            elif not is_gap[y] and in_gap:
                in_gap = False
                gap_len = y - gap_start
                if gap_len > 1:
                    line_gaps.append(float(gap_len))

        line_spacing = float(np.mean(line_gaps)) if line_gaps else 0.0

        return {
            "inter_char_mean": inter_char,
            "inter_char_std": inter_char_std,
            "line_spacing": line_spacing,
        }

    @staticmethod
    def _compute_density_and_sharpness(
        gray: np.ndarray, binary: np.ndarray
    ) -> Dict[str, float]:
        """Compute ink density, edge sharpness, and uniformity."""
        total_pixels = gray.shape[0] * gray.shape[1]
        ink_pixels = np.count_nonzero(binary)
        ink_density = ink_pixels / max(total_pixels, 1)

        # Edge sharpness via Laplacian variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        edge_sharpness = float(lap.var())

        # Text uniformity: how evenly distributed is the ink?
        # Divide into 4x4 grid and compute std of ink density
        h, w = binary.shape
        densities = []
        for row in range(4):
            for col in range(4):
                y0 = row * h // 4
                y1 = (row + 1) * h // 4
                x0 = col * w // 4
                x1 = (col + 1) * w // 4
                cell = binary[y0:y1, x0:x1]
                cell_density = np.count_nonzero(cell) / max(cell.size, 1)
                densities.append(cell_density)

        uniformity = 1.0 - min(float(np.std(densities)) * 5, 1.0)  # normalize

        return {
            "ink_density": ink_density,
            "edge_sharpness": edge_sharpness,
            "uniformity": uniformity,
        }

    def learn_font_profile(
        self,
        text_region: np.ndarray,
        element_class: str = "",
        doc_type: str = "",
    ) -> FontProfile:
        """
        Learn font characteristics from a reference text region.

        Args:
            text_region: Cropped text region image (BGR or grayscale)
            element_class: Element class name (e.g., 'text_name')
            doc_type: Document type (e.g., 'identity')

        Returns:
            FontProfile with learned characteristics
        """
        if text_region is None or text_region.size == 0:
            logger.warning(f"Empty text region for {element_class}")
            return FontProfile(element_class=element_class, doc_type=doc_type)

        gray = (
            cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            if text_region.ndim == 3
            else text_region
        )
        binary = self._to_binary(gray)

        # Compute all features
        stroke = self._compute_stroke_width(binary)
        char_h = self._compute_char_heights(binary)
        spacing = self._compute_spacing(binary)
        density = self._compute_density_and_sharpness(gray, binary)

        # Grayscale histogram (16 bins, normalized)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist = (hist / (hist.sum() + 1e-8)).tolist()

        profile = FontProfile(
            stroke_width_mean=stroke["mean"],
            stroke_width_std=stroke["std"],
            stroke_width_min=stroke["min"],
            stroke_width_max=stroke["max"],
            char_height_mean=char_h["mean"],
            char_height_std=char_h["std"],
            inter_char_spacing_mean=spacing["inter_char_mean"],
            inter_char_spacing_std=spacing["inter_char_std"],
            line_spacing_mean=spacing["line_spacing"],
            ink_density=density["ink_density"],
            edge_sharpness=density["edge_sharpness"],
            text_uniformity=density["uniformity"],
            gray_histogram=hist,
            element_class=element_class,
            doc_type=doc_type,
            image_size=(gray.shape[0], gray.shape[1]),
        )

        logger.info(
            f"  Learned font profile for {element_class}: "
            f"stroke_w={stroke['mean']:.2f}±{stroke['std']:.2f}, "
            f"char_h={char_h['mean']:.1f}, "
            f"density={density['ink_density']:.3f}, "
            f"sharpness={density['edge_sharpness']:.1f}"
        )

        return profile

    def verify_font(
        self,
        text_region: np.ndarray,
        profile: FontProfile,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Verify text region against a learned font profile.

        Args:
            text_region: Cropped text region from submitted document
            profile: Previously learned FontProfile

        Returns:
            (score, details) — score is 0.0-1.0, details dict has per-metric scores
        """
        if text_region is None or text_region.size == 0:
            return 0.0, {"error": "empty region"}

        if profile.stroke_width_mean == 0 and profile.ink_density == 0:
            return 0.8, {"warning": "empty profile, using default score"}

        gray = (
            cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            if text_region.ndim == 3
            else text_region
        )
        binary = self._to_binary(gray)

        # Compute features for the submitted region
        stroke = self._compute_stroke_width(binary)
        char_h = self._compute_char_heights(binary)
        spacing = self._compute_spacing(binary)
        density = self._compute_density_and_sharpness(gray, binary)

        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)

        details: Dict[str, Any] = {}

        # 1. Stroke width similarity (25% weight)
        if profile.stroke_width_mean > 0:
            sw_diff = abs(stroke["mean"] - profile.stroke_width_mean) / max(
                profile.stroke_width_mean, 1e-6
            )
            sw_score = max(0.0, 1.0 - sw_diff)
        else:
            sw_score = 0.8
        details["stroke_width_score"] = round(sw_score, 4)

        # 2. Character height similarity (20% weight)
        if profile.char_height_mean > 0:
            ch_diff = abs(char_h["mean"] - profile.char_height_mean) / max(
                profile.char_height_mean, 1e-6
            )
            ch_score = max(0.0, 1.0 - ch_diff * 0.5)
        else:
            ch_score = 0.8
        details["char_height_score"] = round(ch_score, 4)

        # 3. Ink density similarity (15% weight)
        if profile.ink_density > 0:
            dens_diff = abs(density["ink_density"] - profile.ink_density) / max(
                profile.ink_density, 1e-6
            )
            dens_score = max(0.0, 1.0 - dens_diff)
        else:
            dens_score = 0.8
        details["density_score"] = round(dens_score, 4)

        # 4. Edge sharpness similarity (15% weight)
        if profile.edge_sharpness > 0:
            sharp_diff = abs(
                density["edge_sharpness"] - profile.edge_sharpness
            ) / max(profile.edge_sharpness, 1e-6)
            sharp_score = max(0.0, 1.0 - sharp_diff * 0.3)
        else:
            sharp_score = 0.8
        details["sharpness_score"] = round(sharp_score, 4)

        # 5. Histogram similarity (15% weight)
        ref_hist = np.array(profile.gray_histogram, dtype=np.float32)
        cur_hist = hist.astype(np.float32)
        if ref_hist.sum() > 0:
            hist_score = float(
                cv2.compareHist(ref_hist, cur_hist, cv2.HISTCMP_CORREL)
            )
            hist_score = max(0.0, hist_score)  # CORREL can be [-1, 1]
        else:
            hist_score = 0.8
        details["histogram_score"] = round(hist_score, 4)

        # 6. Text uniformity similarity (10% weight)
        unif_diff = abs(density["uniformity"] - profile.text_uniformity)
        unif_score = max(0.0, 1.0 - unif_diff * 2)
        details["uniformity_score"] = round(unif_score, 4)

        # Weighted final score
        final_score = (
            sw_score * 0.25
            + ch_score * 0.20
            + dens_score * 0.15
            + sharp_score * 0.15
            + hist_score * 0.15
            + unif_score * 0.10
        )

        details["final_score"] = round(final_score, 4)
        return round(final_score, 4), details

    @staticmethod
    def save_profile(profile: FontProfile, path: Path) -> None:
        """Save a font profile to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved font profile: {path}")

    @staticmethod
    def load_profile(path: Path) -> FontProfile:
        """Load a font profile from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FontProfile.from_dict(data)
