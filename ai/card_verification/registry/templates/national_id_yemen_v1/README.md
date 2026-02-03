Template assets

- Place the reference emblem image at: refs/emblem.png
- Place the optional stamp reference image at: refs/stamp.png

Notes
- Both images should be cropped tightly from a valid, high-quality card.
- If stamp.png is not provided, HSV-only detection is expected to be used.

Usage
- Extract reference crops interactively:
  python ai/card_verification/registry/templates/national_id_yemen_v1/tools/extract_refs.py --image "<PATH_TO_CARD_IMAGE>"

Debug ROIs
- Draw template ROIs on a card photo:
  python ai/card_verification/registry/templates/national_id_yemen_v1/tools/debug_rois.py --image "<PATH_TO_CARD_IMAGE>" --template "ai/card_verification/registry/templates/national_id_yemen_v1/layout.yaml"
