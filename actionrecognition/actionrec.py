from mmaction.apis.inferencers import MMAction2Inferencer
from redis import Redis
import numpy as np
import pickle
from argparse import ArgumentParser
from urllib.parse import urlparse
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from operator import itemgetter
import json
def output_result(pred, inference_number, labels):
    print(f'Inference number {inference_number}\n')
    # print(f'length: {len(result)}', flush=True)

    # print(pred, flush=True)
    pred_scores = pred.pred_scores.item.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]
    results = [(labels[k[0]], k[1]) for k in top5_label]
    print('The top-5 labels with corresponding scores are:\n')
    for result in results:
        print(f'{result[0]}: ', result[1])

    return results

def _inputs_to_list(inputs) -> list:
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    return list(inputs)

def get_frame_data(data):
    img = pickle.loads(data[b'image'])
    return img

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input_stream', help='input stream key for coming frames', type=str, default="camera:0")
    parser.add_argument('--output_stream', help='output stream key for action recognition results', type=str,
                        default="camera:0:rec")
    parser.add_argument('--label_file', help='label map file path', type=str)
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--batchSize', type=int, default=1, help='inference batch size')
    parser.add_argument('--sampleSize', type=int, default=10, help='frames sample size')
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://127.0.0.1:6379')

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)
    xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream, conn)

    action_inferencer = MMAction2Inferencer(rec=args.config, device=args.device, input_format='array')
    labels = open(args.label_file).readlines()
    labels = [x.strip() for x in labels]
    if not conn.ping():
        raise Exception('Redis unavailable')

    sampled_frames = []
    inference_count = 1

    last_id, data = xreader_writer.xread_latest_available_message()
    output_id = last_id
    sampled_frames.append(get_frame_data(data))

    while True:
        try:
            ref_id, data = xreader_writer.xread_by_id(last_id)
            if data:
                if len(sampled_frames) < args.sampleSize:
                    sampled_frames.append(get_frame_data(data))
                else:
                    frames_np_array = np.array(sampled_frames)
                    inputs = _inputs_to_list(frames_np_array)
                    preds = action_inferencer.forward(inputs, args.batchSize)

                    results = output_result(preds['rec'][0][0], inference_count, labels)
                    xreader_writer.write_message({'action_rec': json.dumps(results)}, output_id)
                    inference_count = inference_count + 1
                    sampled_frames.clear()
                    sampled_frames.append(get_frame_data(data))
                    output_id = ref_id
                last_id = ref_id
        except ConnectionError as e:
            print("ERROR REDIS CONNECTION: {}".format(e))


if __name__ == "__main__":
    main()
