#!/bin/bash

# Check if correct number of arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <video_base_dir> <output_base_dir>"
    exit 1
fi

# Assign command line arguments to variables
video_base_dir="$1"
output_base_dir="$2"
annotations_file="ucf_temporal_annotations.txt"  # Explicitly set the annotations file name

# Ensure output directory exists
mkdir -p "$output_base_dir"

# Read the annotation file
while IFS= read -r line; do
    read -ra fields <<< "$line"
    filename="${fields[0]}"
    label="${fields[1]}"

    # Proceed only if the label is "Normal"
    if [ "$label" == "Normal" ]; then
        # Find the file in the directory hierarchy
        found_files=$(find "$video_base_dir" -type f -name "$filename")

        if [ -z "$found_files" ]; then
            echo "Warning: File not found - $filename"
            continue
        fi

        video_path=$(echo "$found_files" | head -n 1)

        # Extract every 10th frame as a true negative sample
        ffmpeg -i "$video_path" -vf "select='not(mod(n,10))'" -vsync vfr "$output_base_dir/tn_${filename%.*}_%03d.png"
    fi

done < "$annotations_file"

