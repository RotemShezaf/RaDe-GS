#!/bin/bash
# clear_geodesic_data.sh
# Deletes all geodesic_distance directories under a base folder.
# Usage: ./clear_geodesic_data.sh [base_folder]
# Default base_folder: /home/rotem.shezaf/RaDe-GS/TrainData/Polynomial/SyntheticColmapData/blue_texture

BASE_FOLDER="${1:-/home/rotem.shezaf/RaDe-GS/TrainData/Polynomial/SyntheticColmapData/blue_texture}"

if [[ ! -d "$BASE_FOLDER" ]]; then
    echo "Error: base folder does not exist: $BASE_FOLDER"
    exit 1
fi

echo "Searching for geodesic_distance directories under:"
echo "  $BASE_FOLDER"
echo ""

DIRS=$(find "$BASE_FOLDER" -type d -name "geodesic_distance")
COUNT=$(echo "$DIRS" | grep -c "." || true)

if [[ -z "$DIRS" ]]; then
    echo "No geodesic_distance directories found. Nothing to delete."
    exit 0
fi

echo "Found $COUNT geodesic_distance director(ies):"
echo "$DIRS"
echo ""
read -p "Delete all $COUNT geodesic_distance directories? [y/N] " CONFIRM

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Deleting..."
find "$BASE_FOLDER" -type d -name "geodesic_distance" -exec rm -rf {} +

echo "Done. All geodesic_distance directories have been removed."
