#!/usr/bin/env bash
# Upload local source to Colab, run GPU + regions pipeline, download outputs.
set -euo pipefail

SESSION="${1:-constellatio-regions}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARBALL="/tmp/cn-src-$$.tgz"

cleanup() { colab stop -s "$SESSION" 2>/dev/null || true; }
trap cleanup EXIT

echo "[colab] packing source from $REPO"
tar czf "$TARBALL" -C "$REPO" \
  --exclude='.git' \
  --exclude='algebraics/*.png' \
  --exclude='algebraics/regions/*/*.png' \
  --exclude='algebraics/regions/*/*.pdf' \
  --exclude='algebraics/regions/*/*.npz' \
  --exclude='*.pdf' \
  --exclude='.obsidian' \
  .

echo "[colab] new session $SESSION (T4 GPU)"
colab new -s "$SESSION" --gpu T4

echo "[colab] upload source tarball"
colab upload -s "$SESSION" "$TARBALL" /content/cn-src.tgz

echo "[colab] run pipeline (may take 20–40 min on T4)"
colab exec -s "$SESSION" --timeout 7200 -f "$REPO/algebraics/colab_cli_run.py"

mkdir -p "$REPO/algebraics/regions"
echo "[colab] download outputs"
colab download -s "$SESSION" /content/constellatio-numerorum/algebraics/constellatio.pdf "$REPO/algebraics/constellatio.pdf"
colab download -s "$SESSION" /content/constellatio_regions.zip "$REPO/algebraics/constellatio_regions.zip"

rm -f "$TARBALL"
echo "[colab] done — outputs in algebraics/"
