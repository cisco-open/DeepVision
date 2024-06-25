#!/bin/bash

# Check the number of arguments
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <true_positive_dir> <frame_set_count> <true_positive_count> <true_negative_dir> <true_negative_count> <outputdir>"
    exit 1
fi

# Assign arguments to variables
true_positive_dir="$1"
frame_set_count="$2"
true_positive_count="$3"
true_negative_dir="$4"
true_negative_count="$5"
outputdir="$6"

# Create Output Directories
mkdir -p "$outputdir/true_positives"
mkdir -p "$outputdir/true_negatives"

# Function to Process Directories and Ensure Video Diversity
function process_directory() {
    local src_dir=$1
    local set_count=$2
    local total_sets=$3
    local dest_dir=$4

    declare -A video_counts  # Tracks the number of frames each video has contributed

    # Preprocess to find and shuffle files
    find "$src_dir" -type f -print0 | shuf --zero-terminated | while IFS= read -r -d $'\0' file; do
        local base_name=$(basename "$file")
        local video_name="${base_name%_*}"  # Extract the base video name without frame index

        # Ensure video count is initialized
        if [ -z "${video_counts[$video_name]}" ]; then
            video_counts[$video_name]=0
        fi

        # Check if the video has already contributed the desired number of sets
        if [ "${video_counts[$video_name]}" -ge 1 ]; then
            continue
        fi

        # Select and copy the correct number of consecutive frames
        local frames=($(find "$src_dir" -type f -name "${video_name}_*.png" | sort))
        local num_frames=${#frames[@]}
        if [ "$num_frames" -ge "$set_count" ]; then
            for (( i=0; i<=$num_frames-$set_count; i++ )); do
                local frame_set=("${frames[@]:i:set_count}")
                for frame in "${frame_set[@]}"; do
                    cp "$frame" "$dest_dir"
                done
                ((video_counts[$video_name]++))
                break # Break after processing the first valid set to ensure only one set per video
            done
        fi

        # Check if required sets have been reached
        local total_copied=0
        for count in "${video_counts[@]}"; do
            ((total_copied+=count))
        done

        if [ "$total_copied" -ge "$total_sets" ]; then
            break
        fi
    done
}

# Process True Positives and True Negatives
process_directory "$true_positive_dir" "$frame_set_count" "$true_positive_count" "$outputdir/true_positives"
process_directory "$true_negative_dir" "$frame_set_count" "$true_negative_count" "$outputdir/true_negatives"

echo "Dataset created in '$outputdir'"
