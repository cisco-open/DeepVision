#!/bin/bash

data=$1
embeddings_file=$2

DEFAULT_RESULTS_PATH="./benchmarks"

export EMBEDDINGS=${embeddings_file}

IFS=';' read -ra ADDR <<< "$data"

for i in "${ADDR[@]}"; do
  IFS=',' read -ra VIDEO <<< "$i"
  docker exec -it producer_vision /bin/bash -c "python3 producer.py ${VIDEO[0]} -u redis://redis_vision:6379 --benchmark True --outputFps ${VIDEO[1]}"
  DEFAULT_RESULTS_PATH+="${VIDEO[0]}"
  docker exec -it producer_vision /bin/bash -c "python3 redis_stream_dumper.py --redis redis://redis_vision:6379 --filePath ${DEFAULT_RESULTS_PATH}"
done


