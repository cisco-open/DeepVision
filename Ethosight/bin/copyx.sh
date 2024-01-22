#!/bin/bash

# Check if the right number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./copy_files.sh <number_of_files> <starting_line_number>"
    exit 1
fi

num_files=$1
start_line=$2

rm ../Abuse001_x264/*

# Copy the specified number of files starting from the given line number
for file in $(ls | sort -t "_" -k 2 -n | sed -n "$start_line,$(($start_line + $num_files - 1))p"); do 
    cp $file ../Abuse001_x264 ; 
done

