"""
JSON report generator for verification results.

Produces structured JSON reports with all verification signals,
decision reasoning, and audit trail information.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# JSON Schema for validation
REPORT_SCHEMA = {
    "type": "object",
    "required": [
        "input_file",
        "timestamp",
        "file_hash",
        "logo_hash",
        "verification_results",
        "decision",
        "reasons"
    ],
    "properties": {
        "input_file": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "file_hash": {"type": "string"},
        "logo_hash": {"type": "string"},
        "verification_results": {
            "type": "object",
            "required": [
                "ssim_score",
                "orb_match_ratio",
                "orb_good_matches",
                "resnet_prediction",
                "resnet_confidence"
            ],
            "properties": {
                "ssim_score": {"type": "number", "minimum": -1, "maximum": 1},
                "orb_match_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "orb_good_matches": {"type": "integer", "minimum": 0},
                "resnet_prediction": {"type": "string", "enum": ["genuine", "forged"]},
                "resnet_confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "decision": {"type": "string", "enum": ["AUTHENTIC", "SUSPICIOUS", "FORGED"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "thresholds_used": {"type": "object"}
    }
}


@dataclass
class VerificationResults:
    """Container for verification method results."""
    ssim_score: float
    orb_match_ratio: float
    orb_good_matches: int
    resnet_prediction: str
    resnet_confidence: float
    
    # Siamese
    siamese_score: Optional[float] = None
    siamese_signal: Optional[str] = None
    
    # Optional detailed signals
    ssim_signal: Optional[str] = None
    orb_signal: Optional[str] = None
    resnet_signal: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'ssim_score': round(self.ssim_score, 6),
            'orb_match_ratio': round(self.orb_match_ratio, 6),
            'orb_good_matches': self.orb_good_matches,
            'resnet_prediction': self.resnet_prediction,
            'resnet_confidence': round(self.resnet_confidence, 6),
            'siamese_score': round(self.siamese_score, 6) if self.siamese_score is not None else None,
            'signals': {
                'ssim': self.ssim_signal,
                'orb': self.orb_signal,
                'resnet': self.resnet_signal,
                'siamese': self.siamese_signal
            } if self.ssim_signal else None
        }


@dataclass
class VerificationReport:
    """Complete verification report."""
    input_file: str
    timestamp: str
    file_hash: str
    logo_hash: str
    verification_results: VerificationResults
    decision: str
    reasons: List[str]
    thresholds_used: Optional[Dict] = None
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None
    selected_roi: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        report = {
            'input_file': self.input_file,
            'timestamp': self.timestamp,
            'file_hash': self.file_hash,
            'logo_hash': self.logo_hash,
            'verification_results': self.verification_results.to_dict(),
            'decision': self.decision,
            'reasons': self.reasons
        }
        
        if self.thresholds_used:
            report['thresholds_used'] = self.thresholds_used
        if self.processing_time_ms:
            report['processing_time_ms'] = round(self.processing_time_ms, 2)
        if self.model_version:
            report['model_version'] = self.model_version
        if self.selected_roi:
            report['selected_roi'] = self.selected_roi
        
        return report
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class JSONReporter:
    """Generate JSON reports for verification results."""
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize reporter.
        
        Args:
            thresholds: Threshold configuration to include in reports
        """
        self.thresholds = thresholds
    
    def create_report(
        self,
        input_file: Union[str, Path],
        file_hash: str,
        logo_hash: str,
        ssim_score: float,
        orb_match_ratio: float,
        orb_good_matches: int,
        resnet_prediction: str,
        resnet_confidence: float,
        decision: str,
        reasons: List[str],
        signals: Optional[Dict] = None,
        processing_time_ms: Optional[float] = None,
        siamese_score: Optional[float] = None,
        selected_roi: Optional[Dict] = None
    ) -> VerificationReport:
        """
        Create a verification report.
        
        Args:
            input_file: Path to input image
            file_hash: SHA-256 hash of input file
            logo_hash: SHA-256 hash of cropped logo
            ssim_score: SSIM comparison score
            orb_match_ratio: ORB match ratio
            orb_good_matches: Number of good ORB matches
            resnet_prediction: 'genuine' or 'forged'
            resnet_confidence: Model confidence
            decision: Final decision (AUTHENTIC/SUSPICIOUS/FORGED)
            reasons: List of reasoning explanations
            signals: Optional signal classifications
            processing_time_ms: Optional processing time
            
        Returns:
            VerificationReport object
        """
        results = VerificationResults(
            ssim_score=ssim_score,
            orb_match_ratio=orb_match_ratio,
            orb_good_matches=orb_good_matches,
            resnet_prediction=resnet_prediction,
            resnet_confidence=resnet_confidence,
            ssim_signal=signals.get('ssim') if signals else None,
            orb_signal=signals.get('orb') if signals else None,
            resnet_signal=signals.get('resnet') if signals else None,
            siamese_score=siamese_score,
            siamese_signal=signals.get('siamese') if signals else None
        )
        
        return VerificationReport(
            input_file=str(input_file),
            timestamp=datetime.now().isoformat(),
            file_hash=file_hash,
            logo_hash=logo_hash,
            verification_results=results,
            decision=decision,
            reasons=reasons,
            thresholds_used=self.thresholds,
            processing_time_ms=processing_time_ms,
            selected_roi=selected_roi
        )
    
    def save_report(
        self,
        report: VerificationReport,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Save report to JSON file.
        
        Args:
            report: VerificationReport to save
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        
        return output_path
    
    def validate_report(self, report: Dict) -> bool:
        """
        Validate report against JSON schema.
        
        Args:
            report: Report dictionary to validate
            
        Returns:
            True if valid, raises exception if not
        """
        try:
            import jsonschema
            jsonschema.validate(instance=report, schema=REPORT_SCHEMA)
            return True
        except ImportError:
            # jsonschema not installed, skip validation
            return True
        except jsonschema.ValidationError as e:
            raise ValueError(f"Report validation failed: {e.message}")


def create_batch_report(
    reports: List[VerificationReport],
    output_path: Union[str, Path]
) -> Path:
    """
    Create a batch report from multiple verification results.
    
    Args:
        reports: List of verification reports
        output_path: Output file path
        
    Returns:
        Path to saved file
    """
    batch_report = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(reports),
        'summary': {
            'authentic': sum(1 for r in reports if r.decision == 'AUTHENTIC'),
            'suspicious': sum(1 for r in reports if r.decision == 'SUSPICIOUS'),
            'forged': sum(1 for r in reports if r.decision == 'FORGED')
        },
        'reports': [r.to_dict() for r in reports]
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(batch_report, f, indent=2, ensure_ascii=False)
    
    return output_path


def get_report_schema() -> Dict:
    """Return the JSON schema for validation."""
    return REPORT_SCHEMA
