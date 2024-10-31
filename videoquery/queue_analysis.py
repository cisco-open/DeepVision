import vqpy
from typing import List, Tuple
import ast

from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase
from vqpy.backend.operator import CustomizedVideoReader
from vqpy.utils import NumpyEncoder

from urllib.parse import urlparse
from argparse import ArgumentParser
import json
from redis import Redis
from Monitor import GPUCalculator, MMTMonitor, TSManager
from utils.constants import REDISTIMESERIES, REDISTIMESERIES_PORT, MODEL_RUN_LATENCY, BOUNDING_BOXES_LATENCY, FRAMERATE
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from redis_stream_video_reader import RedisStreamVideoReader
import pickle
import json
from redis import Redis
from Monitor import GPUCalculator, MMTMonitor, TSManager
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from utils.constants import REDISTIMESERIES, REDISTIMESERIES_PORT, MODEL_RUN_LATENCY, BOUNDING_BOXES_LATENCY, FRAMERATE


TOLERANCE = 10


class Person(VObjBase):
    def __init__(self) -> None:
        self.class_name = "person"
        self.object_detector = "yolox"
        self.detector_kwargs = {"device": "gpu"}
        super().__init__()

    @vobj_property(inputs={"tlbr": 0})
    def bottom_center(self, values) -> Tuple[int, int]:
        tlbr = values["tlbr"]
        x = (tlbr[0] + tlbr[2]) / 2
        y = tlbr[3]
        return (x, y)

    @vobj_property(inputs={"tlbr": 0})
    def center(self, values):
        tlbr = values["tlbr"]
        return (tlbr[:2] + tlbr[2:]) / 2

    @vobj_property(inputs={"in_region_frames": 0, "fps": 0})
    def in_region_time(self, values):
        cur_in_region_frames = values["in_region_frames"]
        fps = values["fps"]
        if not cur_in_region_frames:
            return 0
        return round(cur_in_region_frames / fps, 2)

    @vobj_property(inputs={"in_region": 0, "in_region_frames": TOLERANCE})
    def in_region_frames(self, values):
        """
        Return the number of frames that the person is in region continuously.
        If the person is out of region for longer than TOLERANCE, return 0.
        If the person is out of region cur frame and within TORLERANCE,
          the in_region_frames is the same as that of last frame.
        If the person is untracked and tracked again within in TORLENCE frames,
          the time is accumulated. Otherwise, the in_region_frames is 0.
        """
        in_region = values["in_region"]
        # Get the last valid in_region_frames. If person is lost and tracked
        # again, the in_region_frames for lost frames are None.
        last_valid_in_region_frames = 0
        hist_in_region_frames = reversed(values["in_region_frames"][:-1])
        for value in hist_in_region_frames:
            if value is not None:
                last_valid_in_region_frames = value
                break
        if in_region:
            return last_valid_in_region_frames + 1
        else:
            # The person is out of region for longer than TOLERANCE frames
            if last_valid_in_region_frames == values["in_region_frames"][0]:
                return 0
            else:
                return last_valid_in_region_frames

    @vobj_property(inputs={"bottom_center": 0})
    def in_region(self, values):
        bottom_center = values["bottom_center"]
        if bottom_center is not None and vqpy.query.utils.within_regions(
            REGIONS
        )(bottom_center):
            return True
        return False


class QueueAnalysis(QueryBase):
    def __init__(self) -> None:
        self.person = Person()

    def frame_constraint(self):
        return (self.person.in_region_time > 2.7)

    def frame_output(self):
        return (
            self.person.track_id,
            self.person.tlbr,
            self.person.in_region_time
            )


if __name__ == "__main__":
    parser = ArgumentParser("VQPy queue analysis with Deep Vision")
    # parser.add_argument('config', help='config file')
    parser.add_argument('--input_stream', help='input stream key for coming frames', type=str, default="camera:0")
    parser.add_argument('--output_stream', help='output stream key for tracklets', type=str, default="camera:0:vqpy")
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://redis_vision:6379')
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=3000)
    parser.add_argument('--fps', help='Frames per second (webcam)', type=float, default=15.0)
    parser.add_argument(
        "--polygon",
        default=("[(197,158),(544,62),(545,145),(394,247),(315,354)]"),
        help="polygon to define the region of interest",
    )
    args = parser.parse_args()

    url = urlparse(args.redis)
    print(f"using redis url: {url}, maxlen: {args.maxlen}, fps: {args.fps}")
    print(f"host: {url.hostname}, port: {url.port}")
    print(f"polygon: {args.polygon}")
    redis_conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)
    ts_conn = Redis(REDISTIMESERIES, REDISTIMESERIES_PORT)

    xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream, redis_conn, args.maxlen)
    ts_manager = TSManager(ts_conn)
    query_run_monitor = MMTMonitor(ts_manager, ts_key=MODEL_RUN_LATENCY, threshold=15)
    gpu_calculator = GPUCalculator(ts_manager, 15)

    REGIONS = [ast.literal_eval(args.polygon)]

    redis_stream_video_reader = RedisStreamVideoReader(xreader_writer, fps=args.fps)
    query_executor = vqpy.init(
        custom_video_reader=redis_stream_video_reader,
        query_obj=QueueAnalysis(),
        verbose=False,
        additional_frame_fields=["ref_id"],
        output_per_frame_results=True,
    )
    results = vqpy.run(query_executor, print_results=False)

    query_run_monitor.start_timer()
    for result in results:
        query_run_monitor.end_timer()
        ref_id = result.pop("ref_id")
        gpu_calculator.add()
        xreader_writer.write_message({'query': json.dumps(result, cls=NumpyEncoder)}, ref_id)
        query_run_monitor.start_timer()
    query_run_monitor.end_timer()

