#!/bin/bash

# Directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/source/labels" || exit 1

found=false

for file in *.gz; do
    if [ -f "$file" ]; then
        found=true
        echo "Unzipping $file"
        gunzip "$file"
    fi
done

if [ "$found" = false ]; then
    echo "No .gz files found."
else
    echo "Done!"
fi