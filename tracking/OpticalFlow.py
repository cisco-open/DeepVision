import pickle
import cv2
import numpy as np
from redis import Redis
from argparse import ArgumentParser
from urllib.parse import urlparse
from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from utils.Utility import get_frame_data
# from utils.Utility import mls_diff


class OpticalFlow:

    def __init__(self, model: str, device: str,
                 of_rate: int,
                 frame_diff: int,
                 flow_xreader_writer: RedisStreamXreaderWriter,
                 img_xreader_writer: RedisStreamXreaderWriter):

        self.model = init_model(model, device=device)
        self.of_rate = of_rate
        self.frame_diff = frame_diff
        self.flow_xreader_writer = flow_xreader_writer
        self.img_xreader_writer = img_xreader_writer

    def inference(self):
        optical_flow_counter = 0
        frames = []
        last_id, data = self.flow_xreader_writer.xread_latest_available_message()
        output_id = last_id
        frames.append(get_frame_data(data))
        while True:
            if optical_flow_counter % self.of_rate == 0:
                try:
                    ref_id, data = self.flow_xreader_writer.xread_by_id(last_id)
                    frame_diff_counter = 0
                    if len(frames) == 1:
                        while frame_diff_counter != self.frame_diff:
                            ref_id, data = self.flow_xreader_writer.xread_by_id(last_id)
                            if ref_id:
                                last_id = ref_id
                                frame_diff_counter = frame_diff_counter + 1
                    if data:
                        if len(frames) < 2:
                            frames.append(get_frame_data(data))
                        else:
                            result = inference_model(self.model, frames[0], frames[1])
                            flow_map = visualize_flow(result)
                            flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)
                            # frame = np.concatenate((flow_map, frames[0]), axis=1)
                            self.img_xreader_writer.write_message({'img_flow_map': pickle.dumps(flow_map)})
                            self.flow_xreader_writer.write_message({'flow_map': pickle.dumps(result)}, output_id)
                            frames.clear()
                            output_id = last_id
                        last_id = ref_id

                except ConnectionError as e:

                    print("ERROR CONNECTION: {}".format(e))
            optical_flow_counter = optical_flow_counter + 1


def main():
    parser = ArgumentParser()
    parser.add_argument('algo', help='Pretrained action recognition algorithm')
    parser.add_argument('--input_stream', help='input stream key for coming frames', type=str, default="camera:0")
    parser.add_argument('--output_stream_flow', help='output stream key for flow map results', type=str,
                        default="camera:0:flow")
    parser.add_argument('--output_stream_flow_img', help='output stream key for flow map converted to RGB imgs', type=str,
                        default="camera:0:flow_img")
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=3000)
    parser.add_argument('--opticalFlowRate', help='Rate of sampling to inference', type=int, default=5)
    parser.add_argument('--frameDiff', help='Consecutive frames range', type=int, default=0)

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)

    flow_xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream_flow, conn, args.maxlen)
    img_xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream_flow_img, conn, args.maxlen)

    optical_flow = OpticalFlow(args.algo, args.device, args.opticalFlowRate, args.frameDiff, flow_xreader_writer, img_xreader_writer)

    optical_flow.inference()


if __name__ == "__main__":
    main()
