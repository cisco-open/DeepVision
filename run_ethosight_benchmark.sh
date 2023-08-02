#!/bin/bash

data=$1
embeddings_file=$2

DEFAULT_RESULTS_PATH="benchmarks/"
REDIS_URL="redis://redis_vision:6379"

docker exec -itd ethosight_vision /bin/bash -c "source /miniconda/etc/profile.d/conda.sh && conda activate imagebind && python EthosightService.py --redis ${REDIS_URL} --embeddings ${embeddings_file}" 

IFS=';' read -ra ADDR <<< "$data"

for i in "${ADDR[@]}"; do
  IFS=',' read -ra VIDEO <<< "$i"
  docker exec -it producer_vision /bin/bash -c "python3 producer.py ${VIDEO[0]} -u ${REDIS_URL} --benchmark True --outputFps ${VIDEO[1]}"
  filename=$(basename "${VIDEO[0]}")
  filename="${filename%.*}"
  filename+=".json"
  DEFAULT_RESULTS_PATH+="${filename}"
  docker exec -it producer_vision /bin/bash -c "python3 redis_stream_dumper.py --redis ${REDIS_URL} --filePath ${DEFAULT_RESULTS_PATH}"
done


