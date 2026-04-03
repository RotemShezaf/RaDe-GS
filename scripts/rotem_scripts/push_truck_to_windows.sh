#!/bin/bash
# Script to push Truck input.ply from server to Windows computer using rsync
# Run this script on the server

WINDOWS_USER="Rotem Shezaf"
WINDOWS_HOST="212.59.10.131"
SOURCE_FILE="/home/rotem.shezaf/RaDe-GS/data/TNT_GOF/geussians/Truck/input.ply"
DEST_PATH="/cygdrive/c/Users/rotem.shezaf/ron_kimmel/gaussians_files/Truck/"

echo "Pushing file to Windows computer at $WINDOWS_HOST using rsync..."
echo "Source: $SOURCE_FILE"
echo "Destination: $DEST_PATH"

# Use rsync with progress and verbose output
rsync -avz --progress "$SOURCE_FILE" "${WINDOWS_USER}@${WINDOWS_HOST}:${DEST_PATH}"

if [ $? -eq 0 ]; then
    echo "✓ Transfer complete! File saved to Windows"
else
    echo "✗ Transfer failed. Make sure:"
    echo "  1. Your Windows computer has OpenSSH Server installed and running"
    echo "  2. Rsync is available on Windows (install via Cygwin or WSL)"
    echo "  3. You can SSH to your Windows machine: ssh \"${WINDOWS_USER}\"@${WINDOWS_HOST}"
fi
