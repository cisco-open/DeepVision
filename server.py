# RedisEdge realtime video analytics web server
import argparse
import json
import numpy as np
import redis
import pickle 
import cv2
import random
import seaborn as sns

from urllib.parse import urlparse
from PIL import Image
from PIL import ImageDraw
from flask import Flask, Response
from tailvisualization import drawlines, midpointcalculator
from trackletmanager import TrackletManager

UpdatedTracklets = TrackletManager(5)




class RedisImageStream(object):
    def __init__(self, conn, args):
        self.conn = conn
        self.camera = args.camera
        self.boxes = args.boxes
        self.field = args.field.encode('utf-8')

    def random_color(self, object_id):
        """Random a color according to the input seed."""
        random.seed(object_id)
        colors = sns.color_palette().as_hex()
        color = random.choice(colors)
        return color

    def get_last(self):
        """ Gets latest from camera and model """
        p = self.conn.pipeline()
        p.xrevrange(self.camera, count=2)  # Latest {count} frames
        p.xrevrange(self.boxes, count=2)  # Latest {count} tracklets
        frame, tracking_stream = p.execute()

        if tracking_stream and len(tracking_stream[0]) > 0:
            last_frame_refId = tracking_stream[0][1][b'refId'].decode("utf-8")  # Frame reference i
            tracking = json.loads(tracking_stream[0][1][b'tracking'].decode('utf-8'))
            resp = conn.xread({self.camera: last_frame_refId}, count=1)
            key, messages = resp[0]
            frame_last_id, data = messages[0]

            img_data = pickle.loads(data[b'image'])
            label = f'{self.camera}:{frame_last_id}'
            img = Image.fromarray(img_data)

            tracking_info = tracking['tracking_info']
            updated_tracking_info = []
            for tracking_entry in tracking_info:
                objectId = tracking_entry['objectId']
                object_bbox = tracking_entry['object_bbox']
                x1 = object_bbox[0]
                y1 = object_bbox[1]
                x2 = object_bbox[2]
                y2 = object_bbox[3]
                score = object_bbox[4]

                midpoint_list = midpointcalculator(x1,x2,y1,y2)
                midpoint_list.append(score)
                tracking_entry['object_bbox'] = midpoint_list
                updated_tracking_info.append(tracking_entry)
                
                if score > 0.950:
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(((x1, y1), (x2, y2)), width=5, outline=self.random_color(objectId))
                    draw.text(xy=(x1, y1 - 15), text="score: " + str(round(score,3)), fill=self.random_color(objectId))
            tracking['tracking_info'] = updated_tracking_info
            UpdatedTracklets.process_objects(tracking)
            updated_tracklet_values = UpdatedTracklets.values()
            drawlines(updated_tracklet_values, draw)

            arr = np.array(img)
            cv2.putText(arr, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            ret, img = cv2.imencode('.jpg', arr)
            return img.tobytes()

        else:
            print("No tracking info")
            frame_img_data = frame[0][1][b'image']
            img_data = pickle.loads(frame_img_data)
            img = Image.fromarray(img_data)
            arr = np.array(img)
            cv2.putText(arr, 'label', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            ret, img = cv2.imencode('.jpg', arr)
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
    args = parser.parse_args()

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    app.run(host='0.0.0.0', port=5001)
