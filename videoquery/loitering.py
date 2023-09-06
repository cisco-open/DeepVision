import vqpy
from typing import List, Tuple, Dict
import numpy as np
import ast

from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase
from vqpy.backend.operator import CustomizedVideoReader
from vqpy.utils import NumpyEncoder

from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from urllib.parse import urlparse
from argparse import ArgumentParser
from urllib.parse import urlparse
import pickle
import numpy as np
import json
from redis import Redis
from Monitor import GPUCalculator, MMTMonitor, TSManager
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from utils.constants import REDISTIMESERIES, REDISTIMESERIES_PORT, MODEL_RUN_LATENCY, BOUNDING_BOXES_LATENCY, FRAMERATE


NO_RISK = "no_risk"
WARNING = "warning"
ALARM = "alarm"

TOLERANCE = 10

# Construct Person VObj to express VQPy query
class Person(VObjBase):
    def __init__(self) -> None:
        self.class_name = "person"
        self.object_detector = "yolox"
        self.detector_kwargs = {"device": "gpu"}
        super().__init__()

    @vobj_property(inputs={"tlbr": 0})
    def bottom_center(self, values) -> List[Tuple[int, int]]:
        tlbr = values["tlbr"]
        x = (tlbr[0] + tlbr[2]) / 2
        y = tlbr[3]
        return [(x, y)]

    @vobj_property(inputs={"tlbr": 0})
    def center(self, values):
        tlbr = values["tlbr"]
        return (tlbr[:2] + tlbr[2:]) / 2

    @vobj_property(inputs={"in_region_time": 0})
    def loitering(self, values):
        cur_in_region_time = values["in_region_time"]
        if cur_in_region_time >= TIME_ALARM:
            return ALARM
        if cur_in_region_time >= TIME_WARNING:
            return WARNING
        return NO_RISK

    @vobj_property(inputs={"in_region_frames": 0, "fps": 0, "in_region": 0})
    def in_region_time(self, values):
        cur_in_region_frames = values["in_region_frames"]
        fps = values["fps"]
        if not values["in_region"]:
            return 0
        return round(cur_in_region_frames / fps, 2)

    @vobj_property(inputs={"in_region": TOLERANCE, "in_region_frames": TOLERANCE})
    def in_region_frames(self, values):
        """
        Return the number of frames that the person is in region continuously.
        If the person is out of region for longer than TOLERANCE, return 0.
        If the person is out of region cur frame and within TORLERANCE,
          the in_region_frames is the same as that of last frame.
        If the person is untracked and tracked again within in TORLENCE frames,
          the time is accumulated. Otherwise, the in_region_frames is 0.
        """
        in_region_values = values["in_region"]
        # Get the last valid in_region_frames. If person is lost and tracked
        # again, the in_region_frames for lost frames are None.
        last_valid_in_region_frames = 0
        for value in reversed(values["in_region_frames"]):
            if value is not None:
                last_valid_in_region_frames = value
                break
        this_in_region = in_region_values[-1]
        if this_in_region:
            return last_valid_in_region_frames + 1
        else:
            # The person is out of region for longer than TOLERANCE frames
            if last_valid_in_region_frames == in_region_values[0]:
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


# Construct VQPy query
class People_loitering_query(QueryBase):
    def __init__(self) -> None:
        self.person = Person()

    def frame_constraint(self):
        return self.person.in_region_time > 0

    def frame_output(self):
        return (
            self.person.track_id,
            self.person.center,
            self.person.tlbr,
            self.person.loitering,
            self.person.in_region_time,
        )

# Customize a VQPy video reader
class RedisStreamVideoReader(CustomizedVideoReader):
    def __init__(self, 
                 xreader_writer: RedisStreamXreaderWriter,
                 fps: int):
        self.xreader_writer = xreader_writer
        self.fps = fps
        self.first_frame = True
        self.last_id = None
        super().__init__()

    def get_metadata(self) -> Dict:
        return {
            "fps": self.fps,
        }

    def has_next(self) -> bool:
        return True

    def _get_frame_data(self, data):
        frameId = int(data.get(b'frameId').decode())
        img = pickle.loads(data[b'image'])

        return frameId, img

    def _next(self):
        if self.first_frame:
            self.last_id, _ = self.xreader_writer.xread_latest_available_message()
            self.first_frame = False
        while True:
            try:
                ref_id, data = self.xreader_writer.xread_by_id(self.last_id)
                if data:
                    frame_id, image = self._get_frame_data(data)
                    results = {
                        "image": image,
                        "frame_id": frame_id,
                        "ref_id": ref_id,
                    }
                    self.last_id = ref_id
                    return results
            except ConnectionError as e:
                print("ERROR CONNECTION: {}".format(e))


if __name__ == "__main__":
    parser = ArgumentParser("VQPy loitering with Deep Vision")
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
        default=("[(360, 367), (773, 267), (1143, 480), (951, 715), (399, 715)]"),
        help="polygon to define the region of interest",
    )
    parser.add_argument(
        "--time_warning",
        default=4,
        help="time to trigger warning",
    )
    parser.add_argument(
        "--time_alarm",
        default=10,
        help="time to trigger alarm",
    )
    args = parser.parse_args()

    url = urlparse(args.redis)
    print(f"using redis url: {url}, maxlen: {args.maxlen}, fps: {args.fps}")
    print(f"host: {url.hostname}, port: {url.port}")
    redis_conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)
    ts_conn = Redis(REDISTIMESERIES, REDISTIMESERIES_PORT)

    xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream, redis_conn, args.maxlen)
    ts_manager = TSManager(ts_conn)
    query_run_monitor = MMTMonitor(ts_manager, ts_key=MODEL_RUN_LATENCY, threshold=15)
    gpu_calculator = GPUCalculator(ts_manager, 15)

    REGIONS = [ast.literal_eval(args.polygon)]
    TIME_WARNING = args.time_warning
    TIME_ALARM = args.time_alarm

    redis_stream_video_reader = RedisStreamVideoReader(xreader_writer, fps=args.fps)
    query_executor = vqpy.init(
         custom_video_reader=redis_stream_video_reader,
        query_obj=People_loitering_query(),
        verbose=True,
        additional_frame_fields=["ref_id"],
        output_per_frame_results=True,
    )
    results = vqpy.run(query_executor, print_results=False)

    query_run_monitor.start_timer()
    for result in results:
        query_run_monitor.end_timer()
        ref_id = result.pop("ref_id")
        # stream_output format
        # {
        #     "frame_id": 0,
        #     "Person": [
        #           # Person 1
        #           {
        #            "track_id": 1, 
        #            "center": [987.3499755859375, 288.8272399902344],
        #            "tlbr": [929.621826171875, 181.52011108398438, 1045.078125, 396.1343688964844],
        #            "loitering": "alarm",
        #            "in_region_time": 18.6}
        #           # Person 2
        #           {
        #            "track_id": 2,
        #            ...
        #           }
        #         ]
        # }
        gpu_calculator.add()
        xreader_writer.write_message({'query': json.dumps(result, cls=NumpyEncoder)}, ref_id)
        query_run_monitor.start_timer()
    query_run_monitor.end_timer()
