import time
import pickle
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def convert_redis_entry_id_to_mls(entry_id):
    return int(entry_id.split("-")[0])


def diff_since_epoch_mls_with_current(timestamp):
    current_time_ms = int(time.time() * 1000)
    return (current_time_ms - timestamp) / 1000


def is_lt_eq_threshold(entry_id, threshold):
    if not entry_id:
        return False
    id_mls = convert_redis_entry_id_to_mls(entry_id.decode())
    diff = diff_since_epoch_mls_with_current(id_mls)
    return diff < threshold


def get_frame_data(data, key=None):
    if not key:
        result = pickle.loads(data[b'image'])
    elif key == 'frameId':
        result = int(data.get(b'frameId').decode())
    else:
        result = pickle.loads(data[key])
    return result


def get_json_data(data, key):
    return json.loads(data[key].decode('utf-8'))
