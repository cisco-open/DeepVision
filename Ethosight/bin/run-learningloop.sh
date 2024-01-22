#!/bin/bash

if (( $# < 2)); then
  echo "Usage: run-learningloop.sh image_path iterative_hint blank_slate blank_slate_hint"
  echo "Usage example: bash run-learningloop.sh images/shoplifting.png \"retail loss prevention\" "
  echo "Usage example: bash run-learningloop.sh images/shoplifting.png \"retail loss prevention\" true \"shoplifting, person in danger, medical incident in progress, violence of any type\""
  echo "Usage example: bash run-learningloop.sh images/shoplifting.png \"retail loss prevention\" true \"detect normal shopping behavior\""
  echo "Usage example: bash run-learningloop.sh images/shoplifting.png \"retail loss prevention\" true \"detect abnormal shopping behavior\""
  echo "blank_slate is optional, and default as false. If false, prior knowledge from general.embbedings will be used. If true, blank_slate_hint must be provided"

  exit -1
fi

image_path=$1
iterative_hint=$2
blank_slate=${3:-false}
blank_slate_hint=$4

echo "Image path: ${image_path}"
echo "Iterative hint: ${iterative_hint}"
echo "Blank slate: ${blank_slate}"
echo "Blank slate hint: ${blank_slate_hint}"

OUTPUTS_ROOT="outputs"
image_name=$(basename ${image_path%.*})
now=$(date +"%FT%H%M")
outputs_dir="${OUTPUTS_ROOT}/${image_name}/${now}-blankslate-${blank_slate}"

iterations=2

if [ ! -d "${outputs_dir}" ]; then
  mkdir -p "${outputs_dir}"
fi

if [ ${blank_slate} == true ]; then
  echo "Compute label embeddings from blank slate reasoner"

  # write blank slate hint and iterative hint to file
  file="${outputs_dir}/use-case-hints.txt"
  echo "blank_slate_hint: ${blank_slate_hint}" > ${file}
  echo "iterative_hint: ${iterative_hint}" >> ${file}

  else
    echo "Compute label embeddings from general reasoner"
    # write iterative hint to file
    file="${outputs_dir}/use-case-hints.txt"
    echo "iterative_hint:${iterative_hint}" > ${file}
fi

set -o xtrace #echo on
if [ ${blank_slate} == true ]; then
  ./EthosightCLI.py reason --prompt-type blank_slate --use-case "${blank_slate_hint}" -o "${outputs_dir}/iteration0.labels"
  ./EthosightCLI.py embed "${outputs_dir}/iteration0.labels"
  ./EthosightCLI.py affinities "${image_path}" "${outputs_dir}/iteration0.embeddings" --output_filename "${outputs_dir}/iteration0.affinities"
else
  ./EthosightCLI.py affinities "${image_path}" "general.embeddings" --output_filename "${outputs_dir}/iteration0.affinities"
fi
set +o xtrace #echo off

# Iterative learning loop
for (( i=1; i<=iterations; i++ ))
  do
    echo "Iteration ${i}"
    set -o xtrace #echo on
    ./EthosightCLI.py reason --prompt-type iterative --label-affinity-scores "${outputs_dir}/iteration$((i-1)).affinities" --use-case "${iterative_hint}" -o "${outputs_dir}/iteration${i}.labels"
    ./EthosightCLI.py embed "${outputs_dir}/iteration${i}.labels"
    ./EthosightCLI.py affinities "${image_path}" "${outputs_dir}/iteration${i}.embeddings" --output_filename "${outputs_dir}/iteration${i}.affinities"
    rm "${outputs_dir}/iteration${i}.embeddings"
    set +o xtrace #echo off
  done

echo "Outputs are saved to ${outputs_dir}"
