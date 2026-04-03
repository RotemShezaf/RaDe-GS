#!/bin/bash
# Script to pull Truck input.ply from server to Windows computer
# Run this script on your Windows computer (in Git Bash or WSL)

SERVER_USER="rotem.shezaf"
SERVER_HOST="gidpeep"  # Replace with the server address you SSH to
SOURCE_FILE="/home/rotem.shezaf/RaDe-GS/data/TNT_GOF/geussians/Truck/input.ply"
DEST_PATH="/c/Users/rotem.shezaf/ron_kimmel/gaussians_files/Truck/"

# Create destination directory
mkdir -p "$DEST_PATH"

echo "Pulling file from server..."
echo "Source: ${SERVER_USER}@${SERVER_HOST}:${SOURCE_FILE}"
echo "Destination: $DEST_PATH"

# Use rsync to pull the file
rsync -avz --progress "${SERVER_USER}@${SERVER_HOST}:${SOURCE_FILE}" "$DEST_PATH"

if [ $? -eq 0 ]; then
    echo "✓ Transfer complete! File saved to: $DEST_PATH"
else
    echo "✗ Transfer failed."
fi
