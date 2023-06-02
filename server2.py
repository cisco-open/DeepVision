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
import json
import numpy as np
import redis
import pickle 
import cv2
import random
import time
import os
import seaborn as sns

from urllib.parse import urlparse
from PIL import Image
from PIL import ImageDraw
from flask import Flask, Response
from tracklet.tailvisualization import draw_tail, update_midpoint_to_tracklets
from tracklet.trackletmanager import TrackletManager
from dotenv import load_dotenv
import json
import pickle
from typing import List
from utils.DVDisplayChannel import DVDisplayChannel
from pprint import pformat, pprint


updated_tracklets = None

class StreamItem:
    def __init__(self, redis_connection, stream_name):
        self.redis_connection = redis_connection
        self.stream_name = stream_name
        self.ref_id = None
        self.data = None

    def get_last_stream_item(self):
        try:
            resp = self.redis_connection.xrevrange(self.stream_name, count=1)
            if resp:
                self.ref_id, self.data = resp[0]
                return True
            return False
        except redis.exceptions.ConnectionError:
            print("Redis connection error.")
            return False

class VideoFrameStreamItem(StreamItem):
    def __init__(self, redis_connection, stream_name):
        super().__init__(redis_connection, stream_name)

    def get_last_stream_item(self):
        super().get_last_stream_item()
        if self.stream_item:
            self.frame_ref_id, self.data = self.stream_item[0][0]
        else:
            self.frame_ref_id, self.data = None, None

    def get_stream_item(self, frame_ref_id):
        resp = self.redis_connection.xread({self.stream_name: frame_ref_id}, count=1)
        key, messages = resp[0]
        self.frame_ref_id, self.data = messages[0]

    def get_image_data(self):
        if self.frame_ref_id is not None and self.data is not None:
            self.get_stream_item(self.frame_ref_id)
            img_data = pickle.loads(self.data[b'image'])
            label = f'{self.stream_name}:{self.frame_ref_id}'
            return img_data, label
        else:
            return None, None

class TrackingEntry:
    def __init__(self, entry):
        self.objectId = entry['objectId']
        self.object_bbox = entry['object_bbox']
        self.x1 = self.object_bbox[0]
        self.y1 = self.object_bbox[1]
        self.x2 = self.object_bbox[2]
        self.y2 = self.object_bbox[3]
        self.score = self.object_bbox[4]
        self.tracking_entry = entry

    def get(self):
        return self.objectId, self.x1, self.y1, self.x2, self.y2, self.score

class MOTStreamItem(StreamItem):
    def __init__(self, redis_connection, stream_name):
        super().__init__(redis_connection, stream_name)
        self.tracking_info = None

    def get_last_stream_item(self):
        super().get_last_stream_item()
        if self.data:
            self.frame_ref_id = self.data[b'refId'].decode("utf-8")
            self.tracking = json.loads(self.data[b'tracking'].decode('utf-8'))
            self.tracking_info = self.tracking['tracking_info']
            self.tracking_entries = [TrackingEntry(entry) for entry in self.tracking_info]

    def get_tracking_entries(self) -> List[TrackingEntry]:
        return self.tracking_entries


class RedisImageStream(object):
    def __init__(self, conn, args):
        self.conn = conn
        self.pipeline = conn.pipeline()
        self.camera = args.camera
        self.boxes = args.boxes
        self.field = args.field.encode('utf-8')
        self.time = time.time()
        self.trackingstream = MOTStreamItem(self.conn, self.boxes)
        self.videostream = VideoFrameStreamItem(self.conn, self.camera)
        self.ona_display = DVDisplayChannel("ONA_DISPLAY")

    def random_color(self, object_id):
        """Random a color according to the input seed."""
        random.seed(object_id)
        colors = sns.color_palette().as_hex()
        color = random.choice(colors)
        return color

    def get_last(self):
        self.trackingstream.get_last_stream_item()

        if self.trackingstream.frame_ref_id is not None:
            self.videostream.get_stream_item(self.trackingstream.frame_ref_id)

            img_data, label = self.videostream.get_image_data()
            img = Image.fromarray(img_data)
            draw = ImageDraw.Draw(img)

            tracking_enries = self.trackingstream.get_tracking_entries() 
            updated_tracking_info = []
            tail_colors = {}
            for tracking_entry in tracking_enries:
                objectId, x1, y1, x2, y2, score = tracking_entry.get()
                updated_tracking_info.append(update_midpoint_to_tracklets(x1, x2, y1, y2, tracking_entry.tracking_entry))
                
                if score > args.score:
                    tail_colors[objectId] = self.random_color(objectId)
                    draw.rectangle(((x1, y1), (x2, y2)), width=5, outline=tail_colors[objectId])
                    draw.text(xy=(x1, y1 - 15), text="score: " + str(round(score,3)), fill=tail_colors[objectId])

            updated_tracklets.tracklet_collection_for_tail_visualization(updated_tracking_info)
            updated_tracklet_values = updated_tracklets.values()
            draw_tail(updated_tracklet_values, draw, tail_colors)

            arr = np.array(img)
            #cv2.putText(arr, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            msg = None
            msgs = self.ona_display.read_message()
            if msgs is not None:
                msg = msgs[0] if len(msgs) > 0 else None
            x = 10
            y = 40
            if msg is not None:
                message_text = msg.message_text
                label = f"ONA_DISPLAY: {message_text}"
                #text_position = msg.get('text_position', {'x': 0, 'y': 0})
                text_position = msg.text_position
                x = text_position['x']
                y = text_position['y']
                #label = f"ONA_DISPLAY: {pformat(msg)}"
                #pprint(msg)

            cv2.putText(arr, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            ret, img = cv2.imencode('.jpg', arr)
            return img.tobytes()

        else:
            current_time = time.time()
            diff = round(current_time - self.time, 2)
            blank_image = np.zeros((720, 1280, 3), np.uint8)
            model_name = get_model_name()
            cv2.putText(blank_image, f'The tracking model {model_name} is still loading ({diff}s)', (50, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            ret, img = cv2.imencode('.jpg', blank_image)
            return img.tobytes()


def gen(stream):
    while True:
        try:
            frame = stream.get_last()
        except IndexError:
            continue
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Pragma-directive: no-cache\r\n'
               b'Cache-directive: no-cache\r\n'
               b'Cache-control: no-cache\r\n'
               b'Pragma: no-cache\r\n'
               b'Expires: 0\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def get_model_name():
    load_dotenv()
    model_path = os.getenv('MODEL_CONF')
    return model_path.rsplit('/', 1)[-1]


conn = None
args = None
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
   return '<p style="overflow-y: scroll; box-sizing: border-box; margin: 0px; border: 0px; height:600px; width: 1000px;><img src="/video?"></p>'



@app.route('/video')
def video_feed():
    return Response(gen(RedisImageStream(conn, args)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('camera', help='Input camera stream key', nargs='?', type=str, default='camera:0')
    parser.add_argument('boxes', help='Input model stream key', nargs='?', type=str, default='camera:0:mot')
    parser.add_argument('--field', help='Image field name', type=str, default='image')
    parser.add_argument('--fmt', help='Frame storage format', type=str, default='.jpg')
    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--trackletLength', help='Tracklet Length', type=int)
    parser.add_argument('--score', help='Accuracy score treshold', type=float)
    args = parser.parse_args()

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    updated_tracklets = TrackletManager(tracklet_length=args.trackletLength)
    app.run(host='0.0.0.0', port=5001)
