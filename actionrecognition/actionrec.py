from mmaction.apis.inferencers import MMAction2Inferencer
from redis import Redis
import numpy as np
import pickle
import json
from argparse import ArgumentParser
from urllib.parse import urlparse
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from operator import itemgetter
from utils.constants import TOP_LABEL_COUNT, INPUT_FORMAT


class ActionRecognizer:

    def __init__(self, action_inferencer: MMAction2Inferencer,
                 xreader_writer: RedisStreamXreaderWriter,
                 label_file: str,
                 sample_size: int,
                 batch_size: int,
                 top_pred_count: int):
        self.action_inferencer = action_inferencer
        self.xreader_writer = xreader_writer

        labels = open(label_file).readlines()
        self.labels = [x.strip() for x in labels]
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.top_pred_count = top_pred_count
        self.sampled_frames = []

    def _get_frame_data(self, data):
        img = pickle.loads(data[b'image'])
        return img

    def _output_result(self, prediction, labels):
        pred_scores = prediction.pred_scores.item.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        top_label = score_sorted[:self.top_pred_count]

        return [(labels[k[0]], k[1]) for k in top_label]

    def run(self):
        last_id, data = self.xreader_writer.xread_latest_available_message()
        output_id = last_id
        self.sampled_frames.append(self._get_frame_data(data))

        while True:
            try:
                ref_id, data = self.xreader_writer.xread_by_id(last_id)
                if data:
                    if len(self.sampled_frames) < self.sample_size:
                        self.sampled_frames.append(self._get_frame_data(data))
                    else:
                        frames_np_array = np.array(self.sampled_frames)
                        preds = self.action_inferencer.forward(frames_np_array, self.batch_size)

                        results = self._output_result(preds['rec'][0][0], self.labels)
                        self.xreader_writer.write_message({'action_rec': json.dumps(results)}, output_id)
                        self.sampled_frames.clear()
                        self.sampled_frames.append(self._get_frame_data(data))
                        output_id = ref_id
                    last_id = ref_id
            except ConnectionError as e:
                print("ERROR CONNECTION: {}".format(e))


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
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=3000)

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)

    xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream, conn, args.maxlen)

    action_inferencer = MMAction2Inferencer(rec=args.config, device=args.device, input_format=INPUT_FORMAT)

    action_recognizer = ActionRecognizer(action_inferencer, xreader_writer,
                                         args.label_file, args.sampleSize,
                                         args.batchSize, TOP_LABEL_COUNT)

    action_recognizer.run()


if __name__ == "__main__":
    main()
