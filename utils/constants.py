# Tracking module constants
REDISTIMESERIES = "redistimeseries"
REDISTIMESERIES_PORT = 6379
MODEL_RUN_LATENCY = "model_run_latency"
BOUNDING_BOXES_LATENCY = "bounding_boxes_latency"
GPU_METRICS_CMD = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits"
MEMORY_USED = "memory_used"
MEMORY_TOTAL = "memory_total"
GPU_UTILIZATION = "gpu_utilization"
MEMORY_UTILIZATION = "memory_utilization"
GPU_TEMP = "gpu_temp"
FRAMERATE = "framerate"

# Action recognition module constants
TOP_LABEL_COUNT = 3
INPUT_FORMAT = "array"
LABELS_PREFIX = "labels/label_map_"
LABELS_SUFFIX = ".txt"
