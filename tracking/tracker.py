# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from urllib.parse import urlparse
import pickle
from mmtracking.mmtrack.apis import inference_mot, init_model
import numpy as np
import json
import mmcv
from redis import Redis
from Monitor import GPUCalculator, MMTMonitor, TSManager
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from utils.constants import REDISTIMESERIES, REDISTIMESERIES_PORT, MODEL_RUN_LATENCY, BOUNDING_BOXES_LATENCY, FRAMERATE


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Tracker:
    def __init__(self, config_file_path: str,
                 checkpoint_file_path: str,
                 xreader_writer: RedisStreamXreaderWriter,
                 model_run_monitor: MMTMonitor,
                 bbox_run_monitor: MMTMonitor,
                 gpu_calculator: GPUCalculator,
                 ts_manager: TSManager,
                 device: str,
                 class_id: str):

        self.model = init_model(config_file_path, checkpoint_file_path, device)
        self.xreader_writer = xreader_writer
        self.model_run_monitor = model_run_monitor
        self.bbox_run_monitor = bbox_run_monitor
        self.gpu_calculator = gpu_calculator
        self.ts_manager = ts_manager
        self.class_id = class_id

    def _get_frame_data(self, data):
        frameId = int(data.get(b'frameId').decode())
        img = pickle.loads(data[b'image'])

        return frameId, img

    def _construct_response(self, object_ids, bboxes, frame_id, ref_id):
        objects_list = []

        for (i, id) in enumerate(object_ids):
            object_dict = {'objectId': id, 'object_bbox': bboxes[i], 'class': self.class_id}
            objects_list.append(object_dict)
        frame_dict = {'frameId': frame_id, 'tracking_info': objects_list}

        return {'refId': ref_id, 'tracking': json.dumps(frame_dict, cls=NpEncoder)}

    def results2outs(self, bbox_results=None,
                     mask_results=None,
                     mask_shape=None,
                     **kwargs):

        outputs = dict()

        if bbox_results is not None:
            labels = []
            for i, bbox in enumerate(bbox_results):
                labels.extend([i] * bbox.shape[0])
            labels = np.array(labels, dtype=np.int64)
            outputs['labels'] = labels

            bboxes = np.concatenate(bbox_results, axis=0).astype(np.float32)
            if bboxes.shape[1] == 5:
                outputs['bboxes'] = bboxes
            elif bboxes.shape[1] == 6:
                ids = bboxes[:, 0].astype(np.int64)
                bboxes = bboxes[:, 1:]
                outputs['bboxes'] = bboxes
                outputs['ids'] = ids
            else:
                raise NotImplementedError(
                    f'Not supported bbox shape: (N, {bboxes.shape[1]})')

        if mask_results is not None:
            assert mask_shape is not None
            mask_height, mask_width = mask_shape
            mask_results = mmcv.concat_list(mask_results)
            if len(mask_results) == 0:
                masks = np.zeros((0, mask_height, mask_width)).astype(bool)
            else:
                masks = np.stack(mask_results, axis=0)
            outputs['masks'] = masks

        return outputs

    def inference(self):
        last_id, _ = self.xreader_writer.xread_latest_available_message()
        while True:
            try:
                ref_id, data = self.xreader_writer.xread_by_id(last_id)
                if data:
                    frameId, img = self._get_frame_data(data)
                    self.ts_manager.ts_add(FRAMERATE, frameId)

                    self.model_run_monitor.start_timer()
                    result = inference_mot(self.model, img, frame_id=frameId)
                    self.model_run_monitor.end_timer()

                    self.bbox_run_monitor.start_timer()
                    outs_track = self.results2outs(bbox_results=result.get('track_bboxes', None))
                    self.bbox_run_monitor.end_timer()

                    bboxes = outs_track.get('bboxes', None)
                    ids = outs_track.get('ids', None)
                    self.gpu_calculator.add()

                    response = self._construct_response(ids, bboxes, frameId, ref_id)
                    self.xreader_writer.write_message(response)
                    last_id = ref_id
            except ConnectionError as e:
                print("ERROR CONNECTION: {}".format(e))


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input_stream', help='input stream key for coming frames', type=str, default="camera:0")
    parser.add_argument('--classId', help='class category of the objects', type=str, default="PERSON")
    parser.add_argument('--output_stream', help='output stream key for tracklets', type=str, default="camera:0:mot")
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=3000)

    args = parser.parse_args()

    url = urlparse(args.redis)

    redis_conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)
    ts_conn = Redis(REDISTIMESERIES, REDISTIMESERIES_PORT)

    xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream, redis_conn, args.maxlen)
    ts_manager = TSManager(ts_conn)
    model_run_monitor = MMTMonitor(ts_manager, MODEL_RUN_LATENCY, 15)
    bbox_run_latency = MMTMonitor(ts_manager, BOUNDING_BOXES_LATENCY, 15)
    gpu_calculator = GPUCalculator(ts_manager, 15)

    tracker = Tracker(args.config, args.checkpoint,
                      xreader_writer, model_run_monitor,
                      bbox_run_latency, gpu_calculator,
                      ts_manager, args.device, args.classId)

    tracker.inference()
    

if __name__ == "__main__":
    main()



