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
import argparse
import logging
import redis
import time
import pickle
from urllib.parse import urlparse
from videosource import VideoStream


def produce_from_camera():
    stream = VideoStream(isfile=args.webcam, fps=args.outputFps)  # Default to webcam
    for (count, img) in stream:
        msg = {
            'frameId': count,
            'image': pickle.dumps(img)
        }
        _id = conn.xadd(args.output, msg, maxlen=args.maxlen)

    stream.release_camera()


def produce_from_file():
    clear_stream()
    stream = VideoStream(isfile=args.isfile,
                         benchmark=args.benchmark)  # Unless an input file (image or video) was specified
    frame_id = 0  # start new frame count
    rate = stream.video_sample_rate(args.outputFps)
    for (count, img) in stream:
        if count % rate == 0:  # Video fps = 30
            time.sleep(1 / args.outputFps)

            msg = {
                'frameId': frame_id,
                'image': pickle.dumps(img)
            }
            _id = conn.xadd(args.output, msg, maxlen=args.maxlen)
            if args.verbose:
                logging.info('init_frame_count:{}, frame: {} id: {}'.format(count, frame_id, _id))
            frame_id += 1


def clear_stream():
    conn.delete(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('isfile', help='Input file (leave empty to use webcam)', nargs='?', type=str, default=None)
    parser.add_argument('-o', '--output', help='Output stream key name', type=str, default='camera:0')
    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('-w', '--webcam', help='Webcam device number', type=int, default=0)
    parser.add_argument('-v', '--verbose', help='Verbose output', type=bool, default=False)
    # parser.add_argument('--inputFps', help='Frames per second (webcam)', type=float, default=30.0)
    parser.add_argument('--outputFps', help='Frames per second (webcam)', type=float, default=10.0)
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=3000)
    parser.add_argument('--benchmark', help='BenchMark mode or not', type=bool, default=False)

    args = parser.parse_args()

    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port, health_check_interval=25)
    if not conn.ping():
        raise Exception('Redis unavailable')

    if not args.isfile:
        produce_from_camera()
    else:
        produce_from_file()
