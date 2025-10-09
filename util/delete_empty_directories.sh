#!/bin/bash

# Check if the base directory is provided
if [ -z "$1" ]; then
	    echo "Usage: $0 <base_directory>"
	        exit 1
fi

BASE_DIR="$1"

# Check if the provided directory exists
if [ ! -d "$BASE_DIR" ]; then
	    echo "Error: Directory '$BASE_DIR' does not exist."
	        exit 1
fi

# Find and delete empty directories
find "$BASE_DIR" -type d -empty -print -delete

echo "Empty directories deleted from '$BASE_DIR'."
