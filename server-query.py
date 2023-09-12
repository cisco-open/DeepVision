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
from PIL import ImageFont
from flask import Flask, Response
import json
import pickle
from typing import Any, List
from utils.DVDisplayChannel import DVDisplayChannel, DVMessage
from pprint import pformat, pprint
from utils.Utility import is_lt_eq_threshold
from collections import defaultdict

import ast

NO_RISK = "no_risk"
WARNING = "warning"
ALARM = "alarm"

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

    def __init__(self, redis_connection, stream_name, expired_threshold):
        super().__init__(redis_connection, stream_name)
        self.expired_threshold = expired_threshold

    def get_last_stream_item(self):
        super().get_last_stream_item()
        if not is_lt_eq_threshold(self.ref_id, self.expired_threshold):
            self.data = None

    def get_stream_item(self, frame_ref_id):
        resp = self.redis_connection.xread({self.stream_name: frame_ref_id}, count=1, block=None)
        if resp:
            key, messages = resp[0]
            self.ref_id, self.data = messages[0]

    def get_image_data(self):
        if self.ref_id is not None and self.data is not None:
            # self.get_stream_item(self.ref_id)
            img_data = pickle.loads(self.data[b'image'])
            label = f'{self.stream_name}:{self.ref_id}'
            return img_data, label
        else:
            return None, None

class LoiteringQueryEntry:
    def __init__(self, entry):
        self.track_id = entry["track_id"]
        self.center = entry["center"]
        self.in_region_time = entry["in_region_time"]
        self.loitering = entry["loitering"]
        self.object_bbox = entry["tlbr"]
        self.x1 = self.object_bbox[0]
        self.y1 = self.object_bbox[1]
        self.x2 = self.object_bbox[2]
        self.y2 = self.object_bbox[3]

    def get(self):
        return self.in_region_time, self.loitering, self.x1, self.y1, self.x2, self.y2


class QueueQueryEntry:
    def __init__(self, entry):
        self.track_id = entry["track_id"]
        self.in_region_time = entry["in_region_time"]
        self.object_bbox = entry["tlbr"]
        self.x1 = self.object_bbox[0]
        self.y1 = self.object_bbox[1]
        self.x2 = self.object_bbox[2]
        self.y2 = self.object_bbox[3]

    def get(self):
        return self.track_id, self.in_region_time, self.x1, self.y1, self.x2, self.y2


class VideoQueryStreamItem(StreamItem):
    def get_query_data(self):
        if self.data is not None:
            query_results = json.loads(self.data[b'query'].decode('utf-8'))
            frame_id = query_results["frame_id"]
            if args.application == "queue":
                return [QueueQueryEntry(person) for person in query_results["Person"]], frame_id
            elif args.application == "loitering":
                return [LoiteringQueryEntry(person) for person in query_results["Person"]], frame_id
        else:
            return None, None 

    def get_next_stream_item(self):
        if self.ref_id is None:
            return self.get_last_stream_item()
        else:
            resp = self.redis_connection.xread({self.stream_name: self.ref_id}, count=1, block=None)
            if resp:
                key, messages = resp[0]
                self.ref_id, self.data = messages[0]
            return True

class RedisImageStream(object):
    def __init__(self, conn, args):
        self.conn = conn 
        self.pipeline = conn.pipeline()
        self.field = args.field.encode('utf-8')
        self.time = time.time()
        self.query_stream = VideoQueryStreamItem(self.conn, args.query)
        self.videostream = VideoFrameStreamItem(self.conn, args.camera, args.videoThreshold)
        self.message_display = DVDisplayChannel("MESSAGE_DISPLAY")

    def _blank_image(self):
        current_time = time.time()
        diff = round(current_time - self.time, 2)
        blank_image = np.zeros((720, 1280, 3), np.uint8)
        cv2.putText(blank_image, f'The video stream is still loading ({diff}s)', (50, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        ret, img = cv2.imencode('.jpg', blank_image)
        return img.tobytes()

    def random_color(self, object_id):
        """Random a color according to the input seed."""
        random.seed(object_id)
        colors = sns.color_palette().as_hex()
        color = random.choice(colors)
        return color
    
    def get_color(self, object_id):
        colors = sns.color_palette().as_hex()
        color_index = object_id % len(colors)
        return colors[color_index]

    def get_last(self):
        success = self.query_stream.get_next_stream_item()
        if success:
            frame_ref_id = self.query_stream.ref_id
            assert frame_ref_id is not None
            self.videostream.get_stream_item(frame_ref_id)
            img_data, label = self.videostream.get_image_data()
            if img_data is None:
                return self._blank_image()
            img = Image.fromarray(img_data)
            draw = ImageDraw.Draw(img)

            if args.application == "queue":
                draw.polygon(ast.literal_eval(args.polygon), outline='yellow', width=5)
            elif args.application == "loitering":
                draw.polygon(ast.literal_eval(args.polygon), outline='red', width=5)

            query_results, frame_id = self.query_stream.get_query_data()
            aggregrated_query_results = defaultdict(list)
            if query_results is not None:
                for query_entry in query_results:
                    if args.application == "loitering":
                        in_region_time, loitering, x1, y1, x2, y2 = query_entry.get()
                        if loitering == WARNING:
                            color = (0, 165, 255)
                            loitering_message = "Warning: loitering"
                        elif loitering == ALARM:
                            color = (0, 0, 255)
                            loitering_message = "Alarm! Loitering!"
                        else:
                            color = (0, 255, 0)
                        draw.rectangle(((x1, y1), (x2, y2)), width=5, outline=color)
                        # draw in region time at the bottom of the bounding box with color of the bounding box
                        fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
                        draw.text(xy=(x1, y2), text=f"Time in region: {in_region_time}s", fill=color, font=fnt)

                        if loitering == WARNING or loitering == ALARM:
                            # draw loitering message at the top of the bounding box with color of the bounding box
                            fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
                            draw.text(xy=(x1, y1 - 30), text=loitering_message, fill=color, font=fnt)
                    
                    elif args.application == "queue":
                        track_id, in_region_time, x1, y1, x2, y2 = query_entry.get()
                        aggregrated_query_results["in_region_time"].append(in_region_time)
                        color = self.get_color(track_id)
                        draw.rectangle(((x1, y1), (x2, y2)), width=5, outline=color)
                        # draw in region time at the top of the bounding box with color of the bounding box
                        fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
                        text_position = (x1, y1 - 20)
                        text = f"{in_region_time}s"
                        text_bbox = draw.textbbox(text_position, text, font=fnt)
                        draw.rectangle(text_bbox, fill=color)
                        draw.text(xy=text_position, text=text, fill=(255, 255, 255), font=fnt)
            
            # draw aggregration results
            if args.application == "queue":
                in_region_times = aggregrated_query_results["in_region_time"]
                num_in_region = len(in_region_times)
                if num_in_region == 0:
                    text = "No one in the queue"
                else:
                    # draw number of people, min/max/average in region time in the right corner of frame in seperate lines
                    # with font color of red and text background of white and transparency of 50%
                    text = f"Number of people: {num_in_region}\n"
                    average_in_region_time = round(sum(in_region_times) / len(in_region_times), 2)
                    text += f"Average waiting time: {average_in_region_time}s\n"
                    min_in_region_time = round(min(in_region_times), 2)
                    text += f"Min waiting time: {min_in_region_time}s\n"
                    max_in_region_time = round(max(in_region_times), 2)
                    text += f"Max waiting time: {max_in_region_time}s"
                fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
                text_position = (img.width - 320, 0)
                text_bbox = draw.textbbox(text_position, text, font=fnt)
                draw.rectangle(text_bbox, fill=(255, 255, 255, 128))
                draw.text(xy=text_position, text=text, fill=(255, 0, 0), font=fnt)

            arr = np.array(img)
            ret, img = cv2.imencode('.jpg', arr)
            return img.tobytes()
        else:
            return self._blank_image()

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
    parser.add_argument('query', help='Input query stream key', nargs='?', type=str, default='camera:0:vqpy')
    parser.add_argument('--field', help='Image field name', type=str, default='image')
    parser.add_argument('--fmt', help='Frame storage format', type=str, default='.jpg')
    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--videoThreshold', help='Video source stream timing threshold', type=int)
    parser.add_argument("--polygon", default="[(360, 367), (773, 267), (1143, 480), (951, 715), (399, 715)]",
                        help="polygon to define the region of interest", )
    parser.add_argument("--application", choices=["loitering", "queue"], type=str, default="loitering")
    args = parser.parse_args()

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    app.run(host='0.0.0.0', port=5001)
