#!/usr/bin/env bash
# Render all glassbox-llms showcase Manim scenes at high quality.
#
# Usage:
#   ./examples/render_showcase.sh         # high quality (1080p, 60fps)
#   ./examples/render_showcase.sh -ql     # low quality (480p, 15fps) for quick preview
#
# Output lands in media/videos/<scene_name>/
set -euo pipefail

QUALITY="${1:--qh}"  # default to high quality
SCENES_DIR="glassboxllms/visualization/manim_scenes"

echo "=== Rendering glassbox-llms Showcase Scenes ==="
echo "Quality flag: $QUALITY"
echo ""

scenes=(
    "$SCENES_DIR/full_pipeline.py FullPipelineScene"
    "$SCENES_DIR/sae_feature_discovery.py SAEFeatureDiscoveryScene"
    "$SCENES_DIR/probing_hyperplane.py ProbingHyperplaneScene"
    "$SCENES_DIR/steering_vector.py SteeringVectorScene"
    "$SCENES_DIR/circuit_discovery.py CircuitDiscoveryScene"
)

for scene in "${scenes[@]}"; do
    echo "── Rendering: $scene ──"
    manim $QUALITY $scene --disable_caching
    echo ""
done

echo "=== All scenes rendered! ==="
echo "Find videos in: media/videos/"
