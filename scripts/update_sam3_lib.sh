#!/usr/bin/env bash
# Update vendored sam3_lib/ from Facebook's upstream repo.
# Usage: ./scripts/update_sam3_lib.sh

set -euo pipefail

REPO_URL="https://github.com/facebookresearch/sam3.git"
TMPDIR=$(mktemp -d)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
TARGET="$PROJECT_DIR/sam3_lib"

echo "Cloning upstream SAM3 into $TMPDIR ..."
git clone --depth 1 "$REPO_URL" "$TMPDIR/sam3"

echo "Replacing sam3_lib/ ..."
rm -rf "$TARGET"
cp -r "$TMPDIR/sam3/sam3" "$TARGET"

echo "Cleaning up ..."
rm -rf "$TMPDIR"

echo "Done. sam3_lib/ updated from upstream."
echo "Review changes with: git diff --stat sam3_lib/"
