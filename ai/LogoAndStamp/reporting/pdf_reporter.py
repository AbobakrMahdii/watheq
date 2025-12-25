"""
PDF report generator for verification results.

Generates professional PDF reports with:
- Input image preview
- Cropped logo comparison
- Verification scores
- Decision with reasoning
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

try:
    from fpdf import FPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class PDFReporter:
    """Generate PDF reports for verification results."""
    
    def __init__(self):
        """Initialize PDF reporter."""
        if not PDF_SUPPORT:
            raise ImportError(
                "PDF generation requires fpdf2. "
                "Install with: pip install fpdf2"
            )
    
    def create_report(
        self,
        input_file: Union[str, Path],
        input_image: Optional[np.ndarray],
        logo_patch: Optional[np.ndarray],
        reference_logo: Optional[np.ndarray],
        ssim_score: float,
        orb_match_ratio: float,
        orb_good_matches: int,
        resnet_prediction: str,
        resnet_confidence: float,
        decision: str,
        reasons: List[str],
        file_hash: str,
        logo_hash: str,
        output_path: Union[str, Path],
        thresholds: Optional[Dict] = None
    ) -> Path:
        """
        Create a PDF verification report.
        
        Args:
            input_file: Path to input image
            input_image: Input image array (optional, for preview)
            logo_patch: Cropped logo patch (optional)
            reference_logo: Reference logo image (optional)
            ssim_score: SSIM comparison score
            orb_match_ratio: ORB match ratio
            orb_good_matches: Number of good ORB matches
            resnet_prediction: Model prediction
            resnet_confidence: Model confidence
            decision: Final decision
            reasons: List of reasons
            file_hash: File hash
            logo_hash: Logo hash
            output_path: PDF output path
            thresholds: Optional threshold configuration
            
        Returns:
            Path to generated PDF
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 15, 'Wathiq - Logo Verification Report', align='C', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                 align='C', ln=True)
        pdf.ln(10)
        
        # Input file info
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Input Information', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, f'File: {Path(input_file).name}', ln=True)
        pdf.cell(0, 6, f'File Hash: {file_hash[:32]}...', ln=True)
        pdf.cell(0, 6, f'Logo Hash: {logo_hash[:32]}...', ln=True)
        pdf.ln(5)
        
        # Decision (highlighted)
        self._add_decision_box(pdf, decision, reasons)
        pdf.ln(10)
        
        # Verification Scores
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Verification Scores', ln=True)
        pdf.set_font('Helvetica', '', 10)
        
        # Create score table
        self._add_score_table(
            pdf,
            ssim_score,
            orb_match_ratio,
            orb_good_matches,
            resnet_prediction,
            resnet_confidence,
            thresholds
        )
        pdf.ln(10)
        
        # Images section
        if input_image is not None or logo_patch is not None:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Image Analysis', ln=True)
            self._add_images(pdf, input_image, logo_patch, reference_logo, output_path)
        
        # Thresholds if provided
        if thresholds:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, 'Threshold Configuration', ln=True)
            pdf.set_font('Helvetica', '', 9)
            self._add_thresholds_table(pdf, thresholds)
        
        # Save PDF
        pdf.output(str(output_path))
        
        return output_path
    
    def _add_decision_box(
        self,
        pdf: FPDF,
        decision: str,
        reasons: List[str]
    ) -> None:
        """Add decision box with color coding."""
        # Color based on decision
        colors = {
            'AUTHENTIC': (76, 175, 80),    # Green
            'SUSPICIOUS': (255, 193, 7),   # Amber
            'FORGED': (244, 67, 54)        # Red
        }
        color = colors.get(decision, (158, 158, 158))
        
        # Box background
        pdf.set_fill_color(*color)
        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 12, f'  DECISION: {decision}', fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        
        # Reasons
        if reasons:
            pdf.set_font('Helvetica', '', 10)
            pdf.ln(3)
            for i, reason in enumerate(reasons, 1):
                # Truncate long reasons to fit on page
                reason_text = reason[:100] + '...' if len(reason) > 100 else reason
                pdf.cell(0, 5, f'{i}. {reason_text}', ln=True)
    
    def _add_score_table(
        self,
        pdf: FPDF,
        ssim_score: float,
        orb_match_ratio: float,
        orb_good_matches: int,
        resnet_prediction: str,
        resnet_confidence: float,
        thresholds: Optional[Dict]
    ) -> None:
        """Add score table with threshold comparisons."""
        # Table header
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(50, 8, 'Method', border=1, fill=True)
        pdf.cell(40, 8, 'Score', border=1, fill=True, align='C')
        pdf.cell(45, 8, 'Threshold', border=1, fill=True, align='C')
        pdf.cell(45, 8, 'Status', border=1, fill=True, align='C', ln=True)
        
        pdf.set_font('Helvetica', '', 10)
        
        # SSIM row
        ssim_thresh = thresholds.get('ssim', {}).get('strong_genuine', 0.90) if thresholds else 0.90
        ssim_status = 'PASS' if ssim_score >= ssim_thresh else 'FAIL'
        self._add_table_row(pdf, 'SSIM', f'{ssim_score:.4f}', f'>= {ssim_thresh}', ssim_status)
        
        # ORB row
        orb_thresh = thresholds.get('orb', {}).get('strong_genuine', 0.35) if thresholds else 0.35
        orb_status = 'PASS' if orb_match_ratio >= orb_thresh else 'FAIL'
        self._add_table_row(pdf, 'ORB Match Ratio', f'{orb_match_ratio:.4f}', f'>= {orb_thresh}', orb_status)
        
        # ORB matches row
        min_matches = thresholds.get('orb', {}).get('min_good_matches', 10) if thresholds else 10
        matches_status = 'PASS' if orb_good_matches >= min_matches else 'FAIL'
        self._add_table_row(pdf, 'ORB Good Matches', str(orb_good_matches), f'>= {min_matches}', matches_status)
        
        # ResNet row
        resnet_thresh = thresholds.get('resnet', {}).get('strong_genuine', 0.80) if thresholds else 0.80
        if resnet_prediction == 'genuine':
            resnet_status = 'PASS' if resnet_confidence >= resnet_thresh else 'UNCERTAIN'
        else:
            resnet_status = 'FAIL'
        self._add_table_row(
            pdf, 
            f'ResNet ({resnet_prediction})', 
            f'{resnet_confidence:.4f}', 
            f'>= {resnet_thresh}', 
            resnet_status
        )
    
    def _add_table_row(
        self,
        pdf: FPDF,
        method: str,
        score: str,
        threshold: str,
        status: str
    ) -> None:
        """Add a row to the score table."""
        # Status color
        colors = {
            'PASS': (76, 175, 80),
            'FAIL': (244, 67, 54),
            'UNCERTAIN': (255, 193, 7)
        }
        
        pdf.cell(50, 7, method, border=1)
        pdf.cell(40, 7, score, border=1, align='C')
        pdf.cell(45, 7, threshold, border=1, align='C')
        
        # Colored status cell
        color = colors.get(status, (158, 158, 158))
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(45, 7, status, border=1, align='C', fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
    
    def _add_images(
        self,
        pdf: FPDF,
        input_image: Optional[np.ndarray],
        logo_patch: Optional[np.ndarray],
        reference_logo: Optional[np.ndarray],
        output_path: Path
    ) -> None:
        """Add images to the report."""
        temp_dir = output_path.parent / '.temp_images'
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Add input image (resized)
            if input_image is not None:
                input_path = temp_dir / 'input_preview.jpg'
                # Resize for preview
                max_dim = 400
                h, w = input_image.shape[:2]
                scale = min(max_dim / w, max_dim / h)
                if scale < 1:
                    new_size = (int(w * scale), int(h * scale))
                    preview = cv2.resize(input_image, new_size)
                else:
                    preview = input_image
                cv2.imwrite(str(input_path), preview)
                
                pdf.set_font('Helvetica', '', 9)
                pdf.cell(0, 5, 'Input Image:', ln=True)
                pdf.image(str(input_path), x=10, w=80)
                pdf.ln(5)
            
            # Add logo comparison
            if logo_patch is not None:
                logo_path = temp_dir / 'logo_patch.jpg'
                cv2.imwrite(str(logo_path), logo_patch)
                
                pdf.cell(0, 5, 'Extracted Logo:', ln=True)
                pdf.image(str(logo_path), x=10, w=50)
            
            if reference_logo is not None:
                ref_path = temp_dir / 'reference_logo.jpg'
                cv2.imwrite(str(ref_path), reference_logo)
                
                pdf.cell(0, 5, 'Reference Logo:', ln=True)
                pdf.image(str(ref_path), x=70, w=50)
        
        finally:
            # Cleanup temp files
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _add_thresholds_table(self, pdf: FPDF, thresholds: Dict) -> None:
        """Add threshold configuration table."""
        for category, values in thresholds.items():
            if isinstance(values, dict):
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 6, f'{category.upper()}:', ln=True)
                pdf.set_font('Helvetica', '', 9)
                for key, value in values.items():
                    pdf.cell(10, 5, '', ln=False)  # Indent
                    pdf.cell(0, 5, f'{key}: {value}', ln=True)


def generate_pdf_report(
    input_file: Union[str, Path],
    verification_result: Dict,
    output_path: Union[str, Path],
    input_image: Optional[np.ndarray] = None,
    logo_patch: Optional[np.ndarray] = None,
    reference_logo: Optional[np.ndarray] = None
) -> Path:
    """
    Convenience function to generate PDF report from verification result dict.
    
    Args:
        input_file: Input file path
        verification_result: Dict with verification results
        output_path: PDF output path
        input_image: Optional input image array
        logo_patch: Optional cropped logo
        reference_logo: Optional reference logo
        
    Returns:
        Path to generated PDF
    """
    reporter = PDFReporter()
    
    vr = verification_result.get('verification_results', {})
    
    return reporter.create_report(
        input_file=input_file,
        input_image=input_image,
        logo_patch=logo_patch,
        reference_logo=reference_logo,
        ssim_score=vr.get('ssim_score', 0),
        orb_match_ratio=vr.get('orb_match_ratio', 0),
        orb_good_matches=vr.get('orb_good_matches', 0),
        resnet_prediction=vr.get('resnet_prediction', 'unknown'),
        resnet_confidence=vr.get('resnet_confidence', 0),
        decision=verification_result.get('decision', 'UNKNOWN'),
        reasons=verification_result.get('reasons', []),
        file_hash=verification_result.get('file_hash', ''),
        logo_hash=verification_result.get('logo_hash', ''),
        output_path=output_path,
        thresholds=verification_result.get('thresholds_used')
    )
