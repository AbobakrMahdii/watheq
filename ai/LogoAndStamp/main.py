#!/usr/bin/env python3
"""
Wathiq - National ID Logo Verification Pipeline

Main orchestrator that runs the complete verification flow:
1. Load input image (JPG/PNG/PDF)
2. Crop logo ROI
3. Compute hashes
4. Run classical verification (SSIM, ORB)
5. Run deep learning verification (ResNet50)
6. Fuse decisions
7. Generate reports (JSON + optional PDF)

Usage:
    python main.py --input path/to/id_card.jpg --output output/
    python main.py --input path/to/id_card.pdf --pdf --output output/
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('wathiq')


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_verification(
    input_path: str,
    config: dict,
    output_dir: str,
    generate_pdf: bool = False,
    model_path: Optional[str] = None
) -> dict:
    """
    Run the complete verification pipeline.
    
    Args:
        input_path: Path to input image or PDF
        config: Configuration dictionary
        output_dir: Output directory for reports
        generate_pdf: Whether to generate PDF report
        model_path: Optional override for model path
        
    Returns:
        Verification result dictionary
    """
    from utils.image_loader import load_image, preprocess_image
    from utils.roi_cropper import crop_logo_roi, ROIConfig
    from utils.hash_utils import compute_hash, compute_file_hash
    from classical.ssim_matcher import compute_ssim, classify_ssim_signal
    from classical.orb_matcher import match_orb, classify_orb_signal
    from models.resnet_classifier import LogoVerifier
    from fusion.decision_engine import DecisionEngine
    from reporting.json_reporter import JSONReporter, VerificationReport
    
    start_time = time.time()
    
    # Paths
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = config.get('paths', {})
    reference_path = Path(paths.get('reference_logo', 'data/reference/reference_logo.png'))
    model_checkpoint = Path(model_path or paths.get('model_checkpoint', 'models/logo_resnet50.pt'))
    
    thresholds = config.get('thresholds', {})
    preprocessing = config.get('preprocessing', {})
    roi_config = ROIConfig.from_dict(config)
    
    logger.info(f"Processing: {input_path}")
    
    # Step 1: Load input image
    logger.info("Loading input image...")
    try:
        input_image = load_image(input_path)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise
    
    # Step 2: Compute file hash
    logger.info("Computing file hash...")
    file_hash = compute_file_hash(input_path)
    
    # Step 3: Crop logo ROI
    logger.info("Cropping logo region...")
    logo_patch = crop_logo_roi(input_image, roi_config)
    
    # Step 4: Compute logo hash
    logo_hash = compute_hash(logo_patch)
    
    # Step 5: Load reference logo
    logger.info("Loading reference logo...")
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference logo not found: {reference_path}\n"
            "Please place your reference logo at this location."
        )
    reference_logo = load_image(reference_path)
    
    # Step 6: Preprocess for comparison
    target_size = tuple(preprocessing.get('target_size', [224, 224]))
    use_clahe = preprocessing.get('use_clahe', True)
    clahe_clip = preprocessing.get('clahe_clip_limit', 2.0)
    clahe_grid = tuple(preprocessing.get('clahe_tile_grid', [8, 8]))
    
    # Resize reference to match logo patch for comparison
    ref_resized = cv2.resize(reference_logo, (logo_patch.shape[1], logo_patch.shape[0]))
    
    # Step 7: SSIM comparison
    logger.info("Computing SSIM...")
    ssim_score = compute_ssim(
        logo_patch, ref_resized,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip,
        clahe_tile_grid=clahe_grid
    )
    ssim_signal = classify_ssim_signal(ssim_score, thresholds.get('ssim', {}))
    logger.info(f"SSIM Score: {ssim_score:.4f} ({ssim_signal})")
    
    # Step 8: ORB matching
    logger.info("Computing ORB matching...")
    orb_result = match_orb(logo_patch, ref_resized)
    orb_signal = classify_orb_signal(orb_result, thresholds.get('orb', {}))
    logger.info(f"ORB Match Ratio: {orb_result.match_ratio:.4f}, "
                f"Good Matches: {orb_result.num_good_matches} ({orb_signal})")
    
    # Step 9: ResNet prediction
    logger.info("Running ResNet prediction...")
    try:
        verifier = LogoVerifier(model_path=model_checkpoint)
        resnet_pred, resnet_conf = verifier.predict(logo_patch)
        resnet_signal = verifier.classify_signal(
            resnet_pred, resnet_conf, thresholds.get('resnet', {})
        )
        logger.info(f"ResNet: {resnet_pred} @ {resnet_conf:.4f} ({resnet_signal})")
    except FileNotFoundError:
        logger.warning("Model checkpoint not found. Using placeholder prediction.")
        resnet_pred = 'genuine'
        resnet_conf = 0.5
        resnet_signal = 'suspicious'
    
    # Step 10: Decision fusion
    logger.info("Fusing decisions...")
    engine = DecisionEngine(thresholds)
    fusion_result = engine.fuse(
        ssim_score=ssim_score,
        orb_match_ratio=orb_result.match_ratio,
        orb_good_matches=orb_result.num_good_matches,
        resnet_prediction=resnet_pred,
        resnet_confidence=resnet_conf
    )
    
    processing_time = (time.time() - start_time) * 1000  # ms
    
    logger.info(f"Decision: {fusion_result.decision.value}")
    logger.info(f"Processing time: {processing_time:.2f} ms")
    
    # Step 11: Generate JSON report
    logger.info("Generating JSON report...")
    reporter = JSONReporter(thresholds=thresholds)
    
    signals = {
        'ssim': ssim_signal,
        'orb': orb_signal,
        'resnet': resnet_signal
    }
    
    report = reporter.create_report(
        input_file=input_path,
        file_hash=file_hash,
        logo_hash=logo_hash,
        ssim_score=ssim_score,
        orb_match_ratio=orb_result.match_ratio,
        orb_good_matches=orb_result.num_good_matches,
        resnet_prediction=resnet_pred,
        resnet_confidence=resnet_conf,
        decision=fusion_result.decision.value,
        reasons=fusion_result.reasons,
        signals=signals,
        processing_time_ms=processing_time
    )
    
    # Save JSON report
    json_path = output_dir / f"{input_path.stem}_report.json"
    reporter.save_report(report, json_path)
    logger.info(f"JSON report saved: {json_path}")
    
    # Step 12: Generate PDF report (optional)
    if generate_pdf:
        logger.info("Generating PDF report...")
        try:
            from reporting.pdf_reporter import PDFReporter
            pdf_reporter = PDFReporter()
            
            pdf_path = output_dir / f"{input_path.stem}_report.pdf"
            pdf_reporter.create_report(
                input_file=input_path,
                input_image=input_image,
                logo_patch=logo_patch,
                reference_logo=ref_resized,
                ssim_score=ssim_score,
                orb_match_ratio=orb_result.match_ratio,
                orb_good_matches=orb_result.num_good_matches,
                resnet_prediction=resnet_pred,
                resnet_confidence=resnet_conf,
                decision=fusion_result.decision.value,
                reasons=fusion_result.reasons,
                file_hash=file_hash,
                logo_hash=logo_hash,
                output_path=pdf_path,
                thresholds=thresholds
            )
            logger.info(f"PDF report saved: {pdf_path}")
        except ImportError:
            logger.warning("PDF generation requires fpdf2. Skipping PDF report.")
    
    # Print summary
    print("\n" + "=" * 60)
    print(engine.explain_decision(fusion_result))
    print("=" * 60)
    
    return report.to_dict()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Wathiq - National ID Logo Verification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input id_card.jpg
  python main.py --input id_card.pdf --pdf --output results/
  python main.py --input id_card.png --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input ID card image (JPG, PNG, or PDF)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for reports (default: output/)'
    )
    
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Generate PDF report in addition to JSON'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Override model checkpoint path'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Run verification
    try:
        result = run_verification(
            input_path=args.input,
            config=config,
            output_dir=args.output,
            generate_pdf=args.pdf,
            model_path=args.model
        )
        
        # Exit code based on decision
        decision = result.get('decision', 'UNKNOWN')
        if decision == 'AUTHENTIC':
            sys.exit(0)
        elif decision == 'SUSPICIOUS':
            sys.exit(1)
        else:  # FORGED
            sys.exit(2)
            
    except Exception as e:
        logger.exception(f"Verification failed: {e}")
        sys.exit(-1)


if __name__ == '__main__':
    main()
