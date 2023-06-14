from mmaction.apis.inferencers import MMAction2Inferencer
from redis import Redis
import numpy as np
import pickle
from argparse import ArgumentParser
from urllib.parse import urlparse
from utils.RedisStreamXreader import RedisStreamXreader



def get_data_from_resp(resp):
    key, messages = resp[0]
    ref_id, data = messages[0]
    return ref_id, data

def get_frame_data(data):
    img = pickle.loads(data[b'image'])
    return img

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input_stream', help='input stream key for coming frames', type=str, default="camera:0")
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--batchSize', type=int, default=1, help='inference batch size')
    parser.add_argument('--sampleSize', type=int, default=10, help='frames sample size')
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://127.0.0.1:6379')

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)
    xreader = RedisStreamXreader(args.input_stream, conn)

    action_inferencer = MMAction2Inferencer(rec=args.config, device=args.device, input_format='array')

    if not conn.ping():
        raise Exception('Redis unavailable')

    sampled_frames = []

    last_id, data = xreader.xread_latest_available_message()
    sampled_frames.append(get_frame_data(data))

    while True:
        try:
            ref_id, data = xreader.xread_by_id(last_id)
            if data:
                if len(sampled_frames) < args.sampleSize:
                    sampled_frames.append(get_frame_data(data))
                else:
                    frames_np_array = np.array(sampled_frames)
                    result = action_inferencer.forward(frames_np_array, args.batchSize)
                    sampled_frames.clear()
                    print(result, flush=True)
                last_id = ref_id
        except ConnectionError as e:
            print("ERROR REDIS CONNECTION: {}".format(e))


if __name__ == "__main__":
    main()
