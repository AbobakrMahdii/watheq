"""
Decision fusion engine for combining verification signals.

This module implements rule-based decision fusion that combines
classical (SSIM, ORB) and deep learning (ResNet) signals into
a final interpretable decision with explanations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Decision(Enum):
    """Final verification decision."""
    AUTHENTIC = "AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    FORGED = "FORGED"


class SignalStrength(Enum):
    """Individual signal classification."""
    STRONG_GENUINE = "strong_genuine"
    SUSPICIOUS = "suspicious"
    FORGED = "forged"


@dataclass
class VerificationSignals:
    """Container for all verification signals."""
    # SSIM
    ssim_score: float
    ssim_signal: SignalStrength
    
    # ORB
    orb_match_ratio: float
    orb_good_matches: int
    orb_signal: SignalStrength
    
    # ResNet
    resnet_prediction: str  # 'genuine' or 'forged'
    resnet_confidence: float
    resnet_signal: SignalStrength
    
    # Siamese
    siamese_score: float
    siamese_signal: SignalStrength


@dataclass
class FusionResult:
    """Result of decision fusion."""
    decision: Decision
    confidence: float
    reasons: List[str] = field(default_factory=list)
    signals: Optional[VerificationSignals] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'decision': self.decision.value,
            'confidence': self.confidence,
            'reasons': self.reasons,
            'signals': {
                'ssim': {
                    'score': self.signals.ssim_score,
                    'signal': self.signals.ssim_signal.value
                },
                'orb': {
                    'match_ratio': self.signals.orb_match_ratio,
                    'good_matches': self.signals.orb_good_matches,
                    'signal': self.signals.orb_signal.value
                },
                'resnet': {
                    'prediction': self.signals.resnet_prediction,
                    'confidence': self.signals.resnet_confidence,
                    'signal': self.signals.resnet_signal.value
                },
                'siamese': {
                    'score': self.signals.siamese_score,
                    'signal': self.signals.siamese_signal.value
                }
            } if self.signals else None
        }


class DecisionEngine:
    """
    Rule-based decision fusion engine.
    
    Combines SSIM, ORB, and ResNet signals using interpretable rules
    to produce a final verification decision with explanations.
    """
    
    def __init__(self, thresholds: Dict):
        """
        Initialize the decision engine.
        
        Args:
            thresholds: Configuration dict with threshold values
        """
        self.thresholds = thresholds
    
    def classify_ssim(self, score: float) -> SignalStrength:
        """Classify SSIM score into signal strength."""
        ssim_thresh = self.thresholds.get('ssim', {})
        strong = ssim_thresh.get('strong_genuine', 0.90)
        suspicious = ssim_thresh.get('suspicious', 0.70)
        
        if score >= strong:
            return SignalStrength.STRONG_GENUINE
        elif score >= suspicious:
            return SignalStrength.SUSPICIOUS
        else:
            return SignalStrength.FORGED
    
    def classify_orb(
        self, 
        match_ratio: float, 
        good_matches: int
    ) -> SignalStrength:
        """Classify ORB matching result into signal strength."""
        orb_thresh = self.thresholds.get('orb', {})
        strong = orb_thresh.get('strong_genuine', 0.35)
        suspicious = orb_thresh.get('suspicious', 0.15)
        min_matches = orb_thresh.get('min_good_matches', 10)
        
        if match_ratio >= strong and good_matches >= min_matches:
            return SignalStrength.STRONG_GENUINE
        elif match_ratio >= suspicious:
            return SignalStrength.SUSPICIOUS
        else:
            return SignalStrength.FORGED
    
    def classify_resnet(
        self, 
        prediction: str, 
        confidence: float
    ) -> SignalStrength:
        """Classify ResNet prediction into signal strength."""
        resnet_thresh = self.thresholds.get('resnet', {})
        strong = resnet_thresh.get('strong_genuine', 0.85)
        suspicious = resnet_thresh.get('suspicious', 0.50)
        
        if prediction == 'genuine' and confidence >= strong:
            return SignalStrength.STRONG_GENUINE
        elif prediction == 'forged' and confidence >= strong:
            return SignalStrength.FORGED
        else:
            return SignalStrength.SUSPICIOUS

    def classify_siamese(self, score: float) -> SignalStrength:
        """Classify Siamese similarity score."""
        siamese_thresh = self.thresholds.get('siamese', {})
        strong = siamese_thresh.get('strong_genuine', 0.80)
        suspicious = siamese_thresh.get('suspicious', 0.60)
        
        if score >= strong:
            return SignalStrength.STRONG_GENUINE
        elif score >= suspicious:
            return SignalStrength.SUSPICIOUS
        else:
            return SignalStrength.FORGED
    
    def fuse(
        self,
        ssim_score: float,
        orb_match_ratio: float,
        orb_good_matches: int,
        resnet_prediction: str,
        resnet_confidence: float,
        siamese_score: float = 1.0
    ) -> FusionResult:
        """
        Fuse all verification signals into a final decision.
        
        Conservative Rules:
        1. FORGED: ResNet says Forged with high confidence OR 
           (ResNet Forged AND (SSIM Low OR ORB Low OR Siamese Low))
        2. AUTHENTIC: MUST have STRONG_GENUINE from ALL signals.
        3. SUSPICIOUS: Any disagreement or borderline score.
        """
        reasons = []
        
        # Classify individual signals
        ssim_signal = self.classify_ssim(ssim_score)
        orb_signal = self.classify_orb(orb_match_ratio, orb_good_matches)
        resnet_signal = self.classify_resnet(resnet_prediction, resnet_confidence)
        siamese_signal = self.classify_siamese(siamese_score)
        
        # Create signals container
        signals = VerificationSignals(
            ssim_score=ssim_score,
            ssim_signal=ssim_signal,
            orb_match_ratio=orb_match_ratio,
            orb_good_matches=orb_good_matches,
            orb_signal=orb_signal,
            resnet_prediction=resnet_prediction,
            resnet_confidence=resnet_confidence,
            resnet_signal=resnet_signal,
            siamese_score=siamese_score,
            siamese_signal=siamese_signal
        )
        
        # Collect signal states
        all_signals = [ssim_signal, orb_signal, resnet_signal, siamese_signal]
        
        # ===== RULE 1: FORGED DETECTION =====
        # 1a: High confidence ResNet forgery
        if resnet_prediction == 'forged' and resnet_confidence >= 0.85:
            reasons.append(f"ResNet detected forgery with very high confidence ({resnet_confidence:.1%})")
            return FusionResult(Decision.FORGED, resnet_confidence, reasons, signals)
            
        # 1b: ResNet forgery + any classical/siamese forgery
        if resnet_prediction == 'forged':
            if any(s == SignalStrength.FORGED for s in [ssim_signal, orb_signal, siamese_signal]):
                reasons.append("ResNet predicts forged and confirmed by low classical/Siamese similarity")
                return FusionResult(Decision.FORGED, resnet_confidence, reasons, signals)

        # ===== RULE 2: AUTHENTIC DETECTION (STRICT) =====
        if all(s == SignalStrength.STRONG_GENUINE for s in all_signals):
            reasons.append("Verification successful: All signals are strong genuine indicators")
            combined_conf = (ssim_score + orb_match_ratio + resnet_confidence + siamese_score) / 4
            return FusionResult(Decision.AUTHENTIC, combined_conf, reasons, signals)

        # ===== RULE 3: SUSPICIOUS (DEFAULT FALLBACK) =====
        if SignalStrength.FORGED in all_signals:
            reasons.append("Tampering indicators detected by one or more methods")
        elif SignalStrength.SUSPICIOUS in all_signals:
            reasons.append("One or more methods returned borderline/suspicious results")
        
        reasons.append("Conservative classification: Potential risk detected, manual review required")
        return FusionResult(Decision.SUSPICIOUS, 0.5, reasons, signals)
    
    def explain_decision(self, result: FusionResult) -> str:
        """
        Generate a human-readable explanation of the decision.
        
        Args:
            result: FusionResult from fuse()
            
        Returns:
            Formatted explanation string
        """
        lines = [
            f"=== VERIFICATION RESULT: {result.decision.value} ===",
            f"Confidence: {result.confidence:.1%}",
            "",
            "Reasons:"
        ]
        
        for i, reason in enumerate(result.reasons, 1):
            lines.append(f"  {i}. {reason}")
        
        if result.signals:
            lines.extend([
                "",
                "Signal Details:",
                f"  SSIM Score: {result.signals.ssim_score:.4f} ({result.signals.ssim_signal.value})",
                f"  ORB Match Ratio: {result.signals.orb_match_ratio:.4f} ({result.signals.orb_signal.value})",
                f"  ORB Good Matches: {result.signals.orb_good_matches}",
                f"  ResNet: {result.signals.resnet_prediction} @ {result.signals.resnet_confidence:.1%} ({result.signals.resnet_signal.value})",
                f"  Siamese Similarity: {result.signals.siamese_score:.4f} ({result.signals.siamese_signal.value})"
            ])
        
        return "\n".join(lines)


def create_engine(config: Dict) -> DecisionEngine:
    """
    Factory function to create a DecisionEngine.
    
    Args:
        config: Configuration dictionary with 'thresholds' key
        
    Returns:
        Initialized DecisionEngine
    """
    thresholds = config.get('thresholds', {})
    return DecisionEngine(thresholds)
