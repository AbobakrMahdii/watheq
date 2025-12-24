"""
Unit tests for JSON report schema validation.

Tests:
- Report generation produces valid schema
- All required fields are present
- Field types are correct
"""

import json
import pytest
from datetime import datetime

# Try to import jsonschema, skip tests if not available
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

from reporting.json_reporter import (
    JSONReporter,
    VerificationReport,
    VerificationResults,
    REPORT_SCHEMA,
    get_report_schema
)


class TestVerificationResults:
    """Tests for VerificationResults dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        results = VerificationResults(
            ssim_score=0.925,
            orb_match_ratio=0.45,
            orb_good_matches=28,
            resnet_prediction='genuine',
            resnet_confidence=0.87
        )
        
        d = results.to_dict()
        
        assert d['ssim_score'] == 0.925
        assert d['orb_match_ratio'] == 0.45
        assert d['orb_good_matches'] == 28
        assert d['resnet_prediction'] == 'genuine'
        assert d['resnet_confidence'] == 0.87
    
    def test_to_dict_with_signals(self):
        """Test conversion with signal classifications."""
        results = VerificationResults(
            ssim_score=0.92,
            orb_match_ratio=0.40,
            orb_good_matches=25,
            resnet_prediction='genuine',
            resnet_confidence=0.85,
            ssim_signal='strong_genuine',
            orb_signal='strong_genuine',
            resnet_signal='strong_genuine'
        )
        
        d = results.to_dict()
        
        assert d['signals']['ssim'] == 'strong_genuine'
        assert d['signals']['orb'] == 'strong_genuine'
        assert d['signals']['resnet'] == 'strong_genuine'


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""
    
    @pytest.fixture
    def sample_report(self):
        """Create a sample report."""
        results = VerificationResults(
            ssim_score=0.92,
            orb_match_ratio=0.45,
            orb_good_matches=28,
            resnet_prediction='genuine',
            resnet_confidence=0.87
        )
        
        return VerificationReport(
            input_file='test_image.jpg',
            timestamp=datetime.now().isoformat(),
            file_hash='sha256:abc123',
            logo_hash='sha256:def456',
            verification_results=results,
            decision='AUTHENTIC',
            reasons=['All signals agree: strong genuine indicators']
        )
    
    def test_to_dict(self, sample_report):
        """Test conversion to dictionary."""
        d = sample_report.to_dict()
        
        assert d['input_file'] == 'test_image.jpg'
        assert d['decision'] == 'AUTHENTIC'
        assert len(d['reasons']) == 1
        assert 'verification_results' in d
    
    def test_to_json(self, sample_report):
        """Test JSON serialization."""
        json_str = sample_report.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        
        assert parsed['decision'] == 'AUTHENTIC'
        assert parsed['verification_results']['ssim_score'] == 0.92
    
    def test_optional_fields(self, sample_report):
        """Test optional fields are included when present."""
        sample_report.processing_time_ms = 150.5
        sample_report.model_version = '1.0.0'
        
        d = sample_report.to_dict()
        
        assert d['processing_time_ms'] == 150.5
        assert d['model_version'] == '1.0.0'


class TestJSONReporter:
    """Tests for JSONReporter class."""
    
    @pytest.fixture
    def reporter(self):
        """Create a reporter with thresholds."""
        thresholds = {
            'ssim': {'strong_genuine': 0.90, 'suspicious': 0.70},
            'orb': {'strong_genuine': 0.35, 'suspicious': 0.15},
            'resnet': {'strong_genuine': 0.80, 'suspicious': 0.50}
        }
        return JSONReporter(thresholds=thresholds)
    
    def test_create_report(self, reporter):
        """Test report creation."""
        report = reporter.create_report(
            input_file='test.jpg',
            file_hash='sha256:abc',
            logo_hash='sha256:def',
            ssim_score=0.92,
            orb_match_ratio=0.45,
            orb_good_matches=28,
            resnet_prediction='genuine',
            resnet_confidence=0.87,
            decision='AUTHENTIC',
            reasons=['All checks passed']
        )
        
        assert isinstance(report, VerificationReport)
        assert report.decision == 'AUTHENTIC'
    
    def test_report_has_timestamp(self, reporter):
        """Test that report includes timestamp."""
        report = reporter.create_report(
            input_file='test.jpg',
            file_hash='sha256:abc',
            logo_hash='sha256:def',
            ssim_score=0.92,
            orb_match_ratio=0.45,
            orb_good_matches=28,
            resnet_prediction='genuine',
            resnet_confidence=0.87,
            decision='AUTHENTIC',
            reasons=[]
        )
        
        assert report.timestamp is not None
        # Should be ISO format
        datetime.fromisoformat(report.timestamp)
    
    def test_thresholds_included(self, reporter):
        """Test that thresholds are included in report."""
        report = reporter.create_report(
            input_file='test.jpg',
            file_hash='sha256:abc',
            logo_hash='sha256:def',
            ssim_score=0.92,
            orb_match_ratio=0.45,
            orb_good_matches=28,
            resnet_prediction='genuine',
            resnet_confidence=0.87,
            decision='AUTHENTIC',
            reasons=[]
        )
        
        d = report.to_dict()
        assert 'thresholds_used' in d
        assert d['thresholds_used']['ssim']['strong_genuine'] == 0.90


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestSchemaValidation:
    """Tests for JSON schema validation."""
    
    @pytest.fixture
    def valid_report(self):
        """Create a valid report dictionary."""
        return {
            'input_file': 'test.jpg',
            'timestamp': datetime.now().isoformat(),
            'file_hash': 'sha256:abc123',
            'logo_hash': 'sha256:def456',
            'verification_results': {
                'ssim_score': 0.92,
                'orb_match_ratio': 0.45,
                'orb_good_matches': 28,
                'resnet_prediction': 'genuine',
                'resnet_confidence': 0.87
            },
            'decision': 'AUTHENTIC',
            'reasons': ['All checks passed']
        }
    
    def test_valid_report_passes(self, valid_report):
        """Test that valid report passes schema validation."""
        jsonschema.validate(instance=valid_report, schema=REPORT_SCHEMA)
    
    def test_missing_required_field(self, valid_report):
        """Test that missing required field fails validation."""
        del valid_report['decision']
        
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=valid_report, schema=REPORT_SCHEMA)
    
    def test_invalid_decision_value(self, valid_report):
        """Test that invalid decision value fails validation."""
        valid_report['decision'] = 'INVALID'
        
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=valid_report, schema=REPORT_SCHEMA)
    
    def test_invalid_ssim_score(self, valid_report):
        """Test that out-of-range SSIM score fails validation."""
        valid_report['verification_results']['ssim_score'] = 2.0
        
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=valid_report, schema=REPORT_SCHEMA)
    
    def test_invalid_resnet_prediction(self, valid_report):
        """Test that invalid prediction value fails validation."""
        valid_report['verification_results']['resnet_prediction'] = 'unknown'
        
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=valid_report, schema=REPORT_SCHEMA)
    
    def test_negative_orb_matches(self, valid_report):
        """Test that negative match count fails validation."""
        valid_report['verification_results']['orb_good_matches'] = -5
        
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=valid_report, schema=REPORT_SCHEMA)


class TestGetReportSchema:
    """Tests for get_report_schema function."""
    
    def test_returns_schema(self):
        """Test that schema is returned."""
        schema = get_report_schema()
        
        assert isinstance(schema, dict)
        assert 'type' in schema
        assert 'required' in schema
        assert 'properties' in schema
    
    def test_schema_has_required_fields(self):
        """Test that schema defines required fields."""
        schema = get_report_schema()
        
        required = schema['required']
        
        assert 'input_file' in required
        assert 'decision' in required
        assert 'verification_results' in required


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
