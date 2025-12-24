#!/usr/bin/env python3
"""
Wathiq - Hardened Stamp Verification Pipeline

Enhanced orchestrator for stamp verification with:
1. Robust ROI extraction (Jitter Search)
2. Multi-signal analysis: SSIM + ORB + ResNet + Siamese
3. Conservative decision fusion
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('wathiq-stamp')


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_best_stamp_roi(
    input_image: np.ndarray,
    reference_logo: np.ndarray,
    config: dict,
    siamese_verifier=None
) -> Tuple[np.ndarray, dict]:
    """Find the best matching stamp ROI using jitter search."""
    from utils.roi_cropper import StampROIConfig, get_candidate_rois, crop_stamp_roi, evaluate_roi_candidate
    from classical.orb_matcher import match_orb
    
    stamp_config = config.get('stamp_verification', {})
    jitter_config = stamp_config.get('jitter', {'h_range': 0.08, 'v_range': 0.05})
    
    default_roi = StampROIConfig.from_dict(config)
    candidates = get_candidate_rois(
        input_image, 
        default_roi,
        h_jitter=jitter_config.get('h_range', 0.08),
        v_jitter=jitter_config.get('v_range', 0.05)
    )
    
    logger.info(f"Evaluating {len(candidates)} ROI candidates via jitter search...")
    
    best_score = -1.0
    best_roi = None
    best_config = None
    
    for cand_config in candidates:
        patch = crop_stamp_roi(input_image, cand_config)
        score = evaluate_roi_candidate(
            patch, 
            reference_logo, 
            verifier_siamese=siamese_verifier,
            verifier_orb=match_orb
        )
        
        if score > best_score:
            best_score = score
            best_roi = patch
            best_config = cand_config
            
    logger.info(f"Best ROI found with composite score: {best_score:.4f}")
    return best_roi, best_config.to_dict()


def run_verification(
    input_path: str,
    config: dict,
    output_dir: str,
    generate_pdf: bool = False,
    model_path: Optional[str] = None
) -> dict:
    from utils.image_loader import load_image
    from utils.hash_utils import compute_hash, compute_file_hash
    from classical.ssim_matcher import compute_ssim
    from classical.orb_matcher import match_orb
    from models.resnet_classifier import StampVerifier
    from models.siamese_net import SiameseVerifier
    from fusion.decision_engine import DecisionEngine
    from reporting.json_reporter import JSONReporter
    
    start_time = time.time()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = config.get('paths', {})
    reference_path = Path(paths.get('reference_stamp', 'data/reference/reference_stamp.png'))
    model_checkpoint = Path(model_path or paths.get('stamp_model_checkpoint', 'models/stamp_resnet50.pt'))
    
    thresholds = config.get('stamp_verification', {}).get('thresholds', config.get('thresholds', {}))
    
    logger.info(f"STAMP PROCESSING: {input_path}")
    
    # 1. Load images
    input_image = load_image(input_path)
    reference_logo = load_image(reference_path)
    file_hash = compute_file_hash(input_path)
    
    # 2. Robust ROI Search
    siamese_verifier = SiameseVerifier()
    stamp_patch, best_roi_coords = find_best_stamp_roi(
        input_image, reference_logo, config, siamese_verifier
    )
    
    # 3. Compute Hashes
    stamp_hash = compute_hash(stamp_patch)
    
    # 4. Verification Signals
    ref_resized = cv2.resize(reference_logo, (stamp_patch.shape[1], stamp_patch.shape[0]))
    
    # SSIM
    ssim_score = compute_ssim(stamp_patch, ref_resized)
    
    # ORB
    orb_result = match_orb(stamp_patch, ref_resized)
    
    # ResNet
    try:
        resnet_verifier = StampVerifier(model_path=model_checkpoint)
        resnet_pred, resnet_conf = resnet_verifier.predict(stamp_patch)
    except Exception as e:
        logger.warning(f"ResNet failed: {e}. Using placeholder.")
        resnet_pred, resnet_conf = 'forged', 0.5
        
    # Siamese (final score)
    siamese_score = siamese_verifier.compute_similarity(stamp_patch, ref_resized)
    
    # 5. Conservative Decision Fusion
    engine = DecisionEngine(thresholds)
    fusion_result = engine.fuse(
        ssim_score=ssim_score,
        orb_match_ratio=orb_result.match_ratio,
        orb_good_matches=orb_result.num_good_matches,
        resnet_prediction=resnet_pred,
        resnet_confidence=resnet_conf,
        siamese_score=siamese_score
    )
    
    processing_time = (time.time() - start_time) * 1000
    
    # 6. Reporting
    reporter = JSONReporter(thresholds=thresholds)
    signals = {
        'ssim': engine.classify_ssim(ssim_score).value,
        'orb': engine.classify_orb(orb_result.match_ratio, orb_result.num_good_matches).value,
        'resnet': engine.classify_resnet(resnet_pred, resnet_conf).value,
        'siamese': engine.classify_siamese(siamese_score).value
    }
    
    report = reporter.create_report(
        input_file=input_path,
        file_hash=file_hash,
        logo_hash=stamp_hash,
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
    
    # Add stamp-specific details
    report_dict = report.to_dict()
    report_dict['selected_roi'] = best_roi_coords
    report_dict['siamese_score'] = siamese_score
    
    import json
    json_path = output_dir / f"{input_path.stem}_stamp_report.json"
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
        
    logger.info(f"Decision: {fusion_result.decision.value} | Saved: {json_path}")
    
    # Explain
    print("\n" + "=" * 60)
    print(engine.explain_decision(fusion_result))
    print(f"Selected ROI Ratios: {best_roi_coords}")
    print("=" * 60)
    
    return report_dict


def main():
    parser = argparse.ArgumentParser(description='Wathiq - Hardened Stamp Verification')
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    parser.add_argument('--output', '-o', type=str, default='output')
    parser.add_argument('--model', type=str, default=None)
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_verification(args.input, config, args.output, model_path=args.model)


if __name__ == '__main__':
    main()
