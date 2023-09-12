import redis
import argparse
import pickle
import cv2
import numpy as np
from urllib.parse import urlparse
from flask import Flask, Response
from server import RedisImageStream
from server import VideoFrameStreamItem
from server import gen
from server import StreamItem
from utils.Utility import get_frame_data, get_json_data


class RedisFlowImageStream(RedisImageStream):

    def __init__(self, conn, args):
        self.videostream = VideoFrameStreamItem(conn, args.camera, args.videoThreshold)
        self.flow_videostream = StreamItem(conn, args.input_stream_flow_img)
        self.flow_bboxes_stream = StreamItem(conn, args.input_stream_flow)
        self.min_boundary = args.minBoundary
        self.max_boundary = args.maxBoundary

    def _get_blank_yellow_image(self, height, width):
        blank_image = np.zeros((height, width, 3), np.uint8)
        blank_image[:, :] = (255, 255, 0)

        return blank_image

    def _draw_bboxes(self, img_data, bboxes_data):
        contours = get_json_data(bboxes_data, b'flow_map_bboxes')
        for c in contours:

            # if the contour is too small or big, ignore it
            c = np.array(c)
            (x, y, w, h) = cv2.boundingRect(c)
            if (w > self.min_boundary and h > self.min_boundary) and (w < self.max_boundary and h < self.max_boundary):
                cv2.rectangle(img_data, (x, y), (x + w, y + h), (0, 0xFF, 0), 4)

    def get_last(self):
        self.flow_bboxes_stream.get_last_stream_item()
        self.flow_videostream.get_last_stream_item()
        self.videostream.get_last_stream_item()
        video_prev = self.videostream.data.copy()
        img_data = None

        if self.flow_bboxes_stream.data:
            self.videostream.get_stream_item(self.flow_bboxes_stream.ref_id)
            if self.videostream.data:
                img_data = get_frame_data(self.videostream.data)
                self._draw_bboxes(img_data, self.flow_bboxes_stream.data)

        if img_data is None:
            img_data = get_frame_data(video_prev)

        flow_img_data = self._get_blank_yellow_image(*img_data.shape[:2])
        if self.flow_videostream.data:
            flow_img_data = get_frame_data(self.flow_videostream.data, b'img_flow_map')
        frame = np.concatenate((img_data, flow_img_data), axis=1)
        ret, img = cv2.imencode('.jpg', frame)

        return img.tobytes()


conn = None
args = None
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
   return '<p style="overflow-y: scroll; box-sizing: border-box; margin: 0px; border: 0px; height:600px; width: 1000px;><img src="/video?"></p>'



@app.route('/video')
def video_feed():
    return Response(gen(RedisFlowImageStream(conn, args)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('camera', help='Input camera stream key', nargs='?', type=str, default='camera:0')
    parser.add_argument('--input_stream_flow_img', help='stream key for flow map converted to RGB imgs',
                        type=str, default="camera:0:flow_img")

    parser.add_argument('--input_stream_flow', help='stream key for flow bboxes results',
                        type=str, default="camera:0:flow")

    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--videoThreshold', help='Video source stream timing threshold', type=int)
    parser.add_argument('--minBoundary', help='min boundary for height and width of the OF based bbox', type=int)
    parser.add_argument('--maxBoundary', help='max boundary for height and width of the OF based bbox', type=int)

    args = parser.parse_args()

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    app.run(host='0.0.0.0', port=5001)
