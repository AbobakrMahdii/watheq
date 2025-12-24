#!/usr/bin/env python3
"""
Wathiq - Unified Multi-Element Verification Pipeline

Master orchestrator that runs:
1. Logo Verification
2. Stamp Verification
3. (Future) Signature Verification

Produces a consolidated report for the entire document.
"""

import argparse
import logging
import json
import time
from pathlib import Path
import yaml

# Import individual pipelines
import main as logo_pipeline
import main_stamp as stamp_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('wathiq-unified')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_unified_verification(input_path, config, output_dir, generate_pdf=False):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    overall_start = time.time()
    results = {
        "document_info": {
            "filename": input_path.name,
            "timestamp": time.ctime(),
        },
        "verifications": {}
    }

    # 1. Run Logo Verification
    logger.info("--- 🔍 STEP 1: LOGO VERIFICATION ---")
    try:
        logo_results = logo_pipeline.run_verification(
            input_path=str(input_path),
            config=config,
            output_dir=str(output_dir),
            generate_pdf=False
        )
        results["verifications"]["logo"] = logo_results
    except Exception as e:
        logger.warning(f"⚠️ Logo verification skipped or failed: {e}")
        results["verifications"]["logo"] = {"decision": "SKIPPED", "reason": str(e)}

    # 2. Run Stamp Verification
    logger.info("--- 🔍 STEP 2: STAMP VERIFICATION ---")
    try:
        # We need to handle the case where stamp search might fail
        stamp_results = stamp_pipeline.run_verification(
            input_path=str(input_path),
            config=config,
            output_dir=str(output_dir)
        )
        results["verifications"]["stamp"] = stamp_results
    except Exception as e:
        logger.warning(f"⚠️ Stamp verification skipped or failed: {e}")
        results["verifications"]["stamp"] = {"decision": "SKIPPED", "reason": str(e)}

    # Final Decision Logic
    logo_dec = results["verifications"].get("logo", {}).get("decision", "SKIPPED")
    stamp_dec = results["verifications"].get("stamp", {}).get("decision", "SKIPPED")
    
    decisions = [logo_dec, stamp_dec]
    
    if "FORGED" in decisions:
        final_decision = "FORGED"
    elif "SUSPICIOUS" in decisions:
        final_decision = "SUSPICIOUS"
    elif all(d == "SKIPPED" for d in decisions):
        final_decision = "INVALID_DOCUMENT"
    else:
        final_decision = "AUTHENTIC"
        
    results["final_decision"] = final_decision
    results["total_processing_time_ms"] = (time.time() - overall_start) * 1000

    # Save Unified JSON Report
    unified_report_path = output_dir / f"{input_path.stem}_unified_report.json"
    with open(unified_report_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # --- PRETTY PRINT FOR DEMO ---
    print("\n" + "=" * 60)
    print(f"  WATHIQ UNIFIED VERIFICATION REPORT  ".center(60, '='))
    print("=" * 60)
    print(f" File: {input_path.name}")
    
    # Logo Status
    l_status = results["verifications"]["logo"].get("decision", "SKIPPED")
    print(f" [1] Logo Status:  {l_status}")
    
    # Stamp Status
    s_status = results["verifications"]["stamp"].get("decision", "SKIPPED")
    print(f" [2] Stamp Status: {s_status}")
    
    print("-" * 60)
    print(f" FINAL DECISION: {final_decision}".center(60))
    print("=" * 60 + "\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Wathiq Unified Verification')
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    parser.add_argument('--output', '-o', type=str, default='output')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    run_unified_verification(args.input, config, args.output)

if __name__ == "__main__":
    main()
