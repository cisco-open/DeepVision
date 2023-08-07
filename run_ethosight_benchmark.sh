#!/bin/bash

data=$1
embeddings_file=$2

REDIS_URL="redis://redis_vision:6379"
PRODUCING_STREAMS=""
AFF_SCORES_STREAMS=""


IFS=';' read -ra ADDR <<< "$data"

for i in "${ADDR[@]}"; do
  DEFAULT_RESULTS_PATH="benchmarks/"
  IFS=',' read -ra VIDEO <<< "$i"
  filename=$(basename "${VIDEO[0]}")
  filename="${filename%.*}"
  PRODUCING_STREAM="camera:${filename}"
  PRODUCING_STREAMS+="${PRODUCING_STREAM},"
  AFF_SCORES_STREAM="${PRODUCING_STREAM}:affscores"
  AFF_SCORES_STREAMS+="${AFF_SCORES_STREAM},"
  docker exec -itd producer_vision /bin/bash -c "python3 producer.py ${VIDEO[0]} -u ${REDIS_URL} --benchmark True --outputFps ${VIDEO[1]} --output ${PRODUCING_STREAM}"
  filename+=".json"
  DEFAULT_RESULTS_PATH+="${filename}"
  docker exec -itd producer_vision /bin/bash -c "python3 redis_stream_dumper.py --input_stream ${AFF_SCORES_STREAM} --input_stream_original ${PRODUCING_STREAM} --redis ${REDIS_URL} --filePath ${DEFAULT_RESULTS_PATH}"
done
docker exec -it ethosight_vision /bin/bash -c "source /miniconda/etc/profile.d/conda.sh && conda activate imagebind && python EthosightService.py --input_stream ${PRODUCING_STREAMS} --output_stream ${AFF_SCORES_STREAMS}  --redis ${REDIS_URL} --embeddings ${embeddings_file}"


