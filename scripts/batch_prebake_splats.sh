#!/usr/bin/env bash
# Pre-bake the 7 demo template splats sequentially.
# Each template: ~9 min render + ~15s train. Total ~65 min.
set -euo pipefail
cd "$(dirname "$0")/.."

TEMPLATES=(
  eu_de_1bed_munich_schwabing
  in_2bhk_bangalore_vastu
  eu_de_2zimmer_berlin_altbau
  eu_fr_1bed_paris_marais
  gl_jp_1ldk_tokyo_mansion
  gl_us_1bed_atlanta_midtown
  gl_au_2bed_sydney_modern
)

for tid in "${TEMPLATES[@]}"; do
  echo "===================================================================="
  echo "Template: $tid"
  echo "===================================================================="
  if [ -f "/tmp/splat_dataset/$tid/output.ply" ]; then
    echo "  splat exists ($(stat -f%z /tmp/splat_dataset/$tid/output.ply) bytes), skipping"
    continue
  fi
  rm -rf "/tmp/splat_dataset/$tid"
  .venv/bin/python scripts/render_splat_dataset.py "$tid" 2>&1 \
      | grep -vE "^Loading|^/Users|deprecate|warnings.warn|UserWarning" \
      | grep -E "splat|frames|points3D|total" \
      || true
  .venv/bin/python scripts/train_splat.py "$tid" 2>&1 \
      | grep -vE "^Densified|^Opacity|^WARNING" \
      | tail -8
done

echo
echo "===================================================================="
echo "DONE. Splats:"
ls -la /tmp/splat_dataset/*/output.ply 2>/dev/null | awk '{print "  "$NF" "$5" bytes"}'
