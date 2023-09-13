#!/bin/bash

# Execute the appropriate command based on the APPLICATION environment variable
if [[ "$APPLICATION" == "queue" ]]; then
    echo "Starting queue"
    source activate vqpy && python queue_analysis.py --fps "${OUTPUT_FPS}" --redis "${REDIS_URL}" --polygon "${AREA_OF_INTEREST}"
elif [[ "$APPLICATION" == "loitering" ]]; then
    echo "Starting loitering"
    source activate vqpy && python loitering.py --fps "${OUTPUT_FPS}" --redis "${REDIS_URL}" --time_warning "${TIME_WARNING}" --time_alarm "${TIME_ALARM}" --polygon "${AREA_OF_INTEREST}"
fi
