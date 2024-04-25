#!/bin/bash

# This script will find and unzip all .zip files in the current directory

# Loop through each file in the current directory
for file in *.zip
do
    # Check if the file is a zip file
    if [[ "$file" == *.zip ]]
    then
        # Create a directory with the name of the file minus the '.zip' extension
        dir="${file%.zip}"
        mkdir -p "$dir"
        
        # Unzip the file into the directory
        unzip -q "$file" -d "$dir"
        echo "Unzipped $file into $dir/"
    fi
done

echo "Unzipping complete."

