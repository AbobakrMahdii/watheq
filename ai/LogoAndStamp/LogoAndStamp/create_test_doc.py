import cv2
import numpy as np
from pathlib import Path

def create_unified_test_doc():
    # Create a white A4-ish background (1500 height, 1000 width)
    doc = np.ones((1500, 1000, 3), dtype=np.uint8) * 255
    
    # Load references
    logo = cv2.imread('data/reference/reference_logo.png')
    stamp = cv2.imread('data/reference/reference_stamp.png')
    
    if logo is None or stamp is None:
        print("Error: Reference images not found!")
        return
        
    # Resize for placement
    logo = cv2.resize(logo, (200, 200))
    stamp = cv2.resize(stamp, (250, 250))
    
    # Place Logo at top center (y: 50, x: 400)
    doc[50:250, 400:600] = logo
    
    # Place Stamp at bottom center (y: 1100, x: 375)
    doc[1150:1400, 375:625] = stamp
    
    # Save test doc
    output_path = 'unified_test_document.png'
    cv2.imwrite(output_path, doc)
    print(f"Created unified test document: {output_path}")

if __name__ == "__main__":
    create_unified_test_doc()
