#!/usr/bin/env bash
set -euo pipefail

read -r -p "VID= " VID
if [[ -z "${VID}" ]]; then
  echo "VID is required"
  exit 1
fi

IMG="storage/verifications/${VID}/rectify_rebuild/card_rectified.png"
TEMPLATE="ai/card_verification/registry/templates/national_id_yemen_v1/layout.yaml"
OUT_DIR="storage/verifications/${VID}/layout_rebuild"
PICKER="ai/card_verification/registry/templates/national_id_yemen_v1/tools/pick_roi.py"

if [[ ! -f "${IMG}" ]]; then
  echo "Image not found: ${IMG}"
  exit 1
fi
if [[ ! -f "${PICKER}" ]]; then
  echo "ROI picker not found: ${PICKER}"
  exit 1
fi

elements=(
  emblem
  photo
  stamp_under_photo_band
  stamp_expected
  name_text
  national_id_number
)

for element in "${elements[@]}"; do
  echo "Selecting ROI for: ${element}"
  output="$(python "${PICKER}" --image "${IMG}" --template "${TEMPLATE}" --element "${element}")"
  echo "${output}"

  line="$(echo "${output}" | grep -E "ROI ratios:\s*\{x:")"
  if [[ -z "${line}" ]]; then
    echo "Failed to parse ROI ratios for ${element}"
    exit 1
  fi

  x="$(echo "${line}" | sed -E 's/.*x:[[:space:]]*([0-9.]+).*/\1/')"
  y="$(echo "${line}" | sed -E 's/.*y:[[:space:]]*([0-9.]+).*/\1/')"
  w="$(echo "${line}" | sed -E 's/.*w:[[:space:]]*([0-9.]+).*/\1/')"
  h="$(echo "${line}" | sed -E 's/.*h:[[:space:]]*([0-9.]+).*/\1/')"

  python scripts/set_roi.py "${element}" "${x}" "${y}" "${w}" "${h}"
done

python ai/card_verification/layout_verify.py \
  --image "${IMG}" \
  --rectified "${IMG}" \
  --template "${TEMPLATE}" \
  --out_dir "${OUT_DIR}"
