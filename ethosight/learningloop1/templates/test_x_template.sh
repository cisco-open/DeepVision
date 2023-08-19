#!/bin/bash

# Get the current script's directory
script_dir=$(dirname "$0")

# Extract the base name of the script without extension
base_name=$(basename "$0" .sh)

# Extract the last part of the current working directory
current_immediate_dir=${PWD##*/}

output_file="${script_dir}/${current_immediate_dir}/${base_name}_output.txt"

# Change directory to one level up
cd ..

# Construct the desired path
desired_path="$script_dir/$current_immediate_dir"

# Run the program and save the output
EthosightAppCLI optimize "$desired_path" | tee "$output_file"
