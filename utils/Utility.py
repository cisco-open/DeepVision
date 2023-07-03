import time
import pickle


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


def get_frame_data(data):
    img = pickle.loads(data[b'image'])
    return img
