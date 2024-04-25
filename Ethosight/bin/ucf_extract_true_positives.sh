#!/bin/bash

# Check if correct number of arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <video_base_dir> <output_base_dir>"
    exit 1
fi

# Assign command line arguments to variables
video_base_dir="$1"
output_base_dir="$2"

# Ensure output directory exists
mkdir -p "$output_base_dir"

# Assuming annotations are in the current directory named 'ucf_temporal_annotations.txt'
while IFS= read -r line; do
    # Read fields separated by space
    read -ra fields <<< "$line"
    filename="${fields[0]}"
    start_frame="${fields[2]}"
    end_frame="${fields[3]}"

    # Find the file in the directory hierarchy
    found_files=$(find "$video_base_dir" -type f -name "$filename")

    # Check if file was found
    if [ -z "$found_files" ]; then
        echo "Warning: File not found - $filename"
        continue
    fi

    # Assuming only one file will match, or handling the first match
    video_path=$(echo "$found_files" | head -n 1)

    # Extract frames if the start and end frames are not -1
    if [ "$start_frame" -ne -1 ] && [ "$end_frame" -ne -1 ]; then
        ffmpeg -i "$video_path" -vf "select='not(mod(n,10))*between(n,$start_frame,$end_frame)'" -vsync vfr "$output_base_dir/${filename%.*}_${start_frame}_to_${end_frame}_%03d.png"
    fi

    # Check if there is a second event; fields are indexed starting from 4 and 5
    event2_start="${fields[4]}"
    event2_end="${fields[5]}"
    if [ "$event2_start" -ne -1 ] && [ "$event2_end" -ne -1 ]; then
        ffmpeg -i "$video_path" -vf "select='not(mod(n,10))*between(n,$event2_start,$event2_end)'" -vsync vfr "$output_base_dir/${filename%.*}_${event2_start}_to_${event2_end}_%03d.png"
    fi

done < "ucf_temporal_annotations.txt"

