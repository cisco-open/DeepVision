import pickle
import cv2
import time
import json
import numpy as np
from redis import Redis
from argparse import ArgumentParser
from urllib.parse import urlparse
from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from utils.Utility import get_frame_data
from Monitor import TSManager, GPUCalculator, MMTMonitor
from tracker import NpEncoder
from utils.constants import FRAMERATE, FLOW_RUN_LATENCY, VISUAlIZE_RUN_LATENCY, REDISTIMESERIES, REDISTIMESERIES_PORT


class OpticalFlow:

    def __init__(self, model: str, device: str,
                 of_rate: int,
                 frame_diff: int,
                 threshold: int,
                 flow_xreader_writer: RedisStreamXreaderWriter,
                 img_xreader_writer: RedisStreamXreaderWriter,
                 model_run_monitor: MMTMonitor,
                 visualize_run_monitor: MMTMonitor,
                 gpu_calculator: GPUCalculator,
                 ts_manager: TSManager):

        self.model = init_model(model, device=device)
        self.of_rate = of_rate
        self.frame_diff = frame_diff
        self.threshold = threshold
        self.flow_xreader_writer = flow_xreader_writer
        self.img_xreader_writer = img_xreader_writer
        self.model_run_monitor = model_run_monitor
        self.visualize_run_monitor = visualize_run_monitor
        self.gpu_calculator = gpu_calculator
        self.ts_manager = ts_manager

    def hsv(self, flow):

        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
        hsv[..., 1] = 255

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv[..., 0] = angle * 180 / np.pi / 2

        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        hsv = np.asarray(hsv, dtype=np.uint8)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def _handle_bboxes(self, flow):
        gray = cv2.cvtColor(self.hsv(flow), cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, self.threshold, 255,
                               cv2.THRESH_BINARY)[1]

        cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        return cnts

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
                            self.model_run_monitor.start_timer()
                            result = inference_model(self.model, frames[0], frames[1])
                            cnts = self._handle_bboxes(result)
                            self.model_run_monitor.end_timer()

                            self.visualize_run_monitor.start_timer()
                            flow_map = visualize_flow(result)
                            self.visualize_run_monitor.end_timer()

                            self.gpu_calculator.add()
                            flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)
                            self.img_xreader_writer.write_message({'img_flow_map': pickle.dumps(flow_map)})
                            self.flow_xreader_writer.write_message({'flow_map_bboxes': json.dumps(cnts, cls=NpEncoder)},
                                                                   output_id)
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
    parser.add_argument('--grayScaleThreshold', help='Threshold value for gray scale thresholding', type=int, default=10)

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)
    ts_conn = Redis(REDISTIMESERIES, REDISTIMESERIES_PORT)
    ts_manager = TSManager(ts_conn)
    model_run_monitor = MMTMonitor(ts_manager, FLOW_RUN_LATENCY, 15)
    visualize_run_monitor = MMTMonitor(ts_manager, VISUAlIZE_RUN_LATENCY, 15)
    gpu_calculator = GPUCalculator(ts_manager, 15)

    flow_xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream_flow, conn, args.maxlen + 5000)
    img_xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream_flow_img, conn, args.maxlen)

    optical_flow = OpticalFlow(args.algo, args.device, args.opticalFlowRate,
                               args.frameDiff, args.grayScaleThreshold,
                               flow_xreader_writer, img_xreader_writer, model_run_monitor,
                               visualize_run_monitor, gpu_calculator, ts_manager)

    optical_flow.inference()


if __name__ == "__main__":
    main()
