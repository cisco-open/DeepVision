import argparse
from asyncio import sleep
import json
import redis
import logging

from redistimeseries.client import Client as RedisTimeSeries
from tracklet.tsconversion import aspect_ratio
from urllib.parse import urlparse
from tracklet.trackletmanager import TrackletManager



class TrackertoTimeSeries(object):
    """Create Redis TimeSeries stream entries for each object

        Args:
        conn (stream): Redis connection (read)
        ts_conn (redistimeseries) : Redis connection (write)
        boxes (key): Redis input stream key contains object tracking infos {default='camera:0:mot'}
        ts (conn): Redis timeseries connection
        last_process_Id (str): Last processed FrameId
    """

    def __init__(self, conn, ts_conn, args):
        self.conn = conn
        self.ts_conn = ts_conn
        self.ts = RedisTimeSeries(conn=ts_conn)
        self.last_process_Id = ""
        self.boxes = args.boxes
        self.max_skipped_frame_allowed = args.max_skipped_frame_allowed
        self.tracklet_manager = TrackletManager(self.max_skipped_frame_allowed)


    def process_statuses(self, statuses):
        """ Label timeseries objects statuses

        Args:
            statuses (dict): Stored statuses of objects
        """
        for temp_object_id, status in list(statuses.items()):
            if status == TrackletManager.STATUS_ACTIVE:
                self.ts.alter(self.boxes + ":asr:" + f"{temp_object_id}", labels={'status': 'active'})
            elif status == TrackletManager.STATUS_INACTIVE:
                self.ts.alter(self.boxes + ":asr:" + f"{temp_object_id}", labels={'status': 'inactive'})
            else:
                pass


    def get_last(self):
        """ Gets latest from mmtracker tracking info convert to timeseries
        """
        p = self.conn.pipeline()
        p.xrevrange(self.boxes, count=1) 
        tracking_stream = p.execute()

        if tracking_stream and len(tracking_stream[0]) > 0 : 
            last_mmtracking_id = tracking_stream[0][0][0].decode("utf-8")
            tracking = json.loads(tracking_stream[0][0][1][b'tracking'].decode('utf-8'))
            tracking_info = tracking["tracking_info"]
            frameId = tracking["frameId"]

            # Objects in one frame.
            objects_dict = {}  
            if frameId != self.last_process_Id:
                self.last_process_Id = frameId
                
                for tracking_entry in tracking_info:
                    objectID = tracking_entry['objectId']

                    # Get objects bbox info
                    object_bbox = tracking_entry['object_bbox']
                    objects_dict[objectID] = object_bbox

                    x1 = object_bbox[0]
                    y1 = object_bbox[1]
                    x2 = object_bbox[2]
                    y2 = object_bbox[3]

                    # aspect ratio calculation
                    ratio = aspect_ratio(x1, y1, x2, y2)
                    # timeseries timestamp format.
                    res_msec = int(str(last_mmtracking_id).split('-')[0])

                    # Create / Add timeseries for each object in frame.
                    self.ts.add(self.boxes + ":asr:" + f"{objectID}", res_msec, ratio)

                # Store statuses of objects
                statuses = self.tracklet_manager.process_objects(objects_dict)
                self.process_statuses(statuses)
        
            return True
        return False

def gen(stream):
    
    while True:
       
        frame = stream.get_last()
        if frame is False:
            logging.info("waiting!")
            sleep(0.5)
                      
                
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('boxes', help='Input model stream key', nargs='?', type=str, default='camera:0:mot')
    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://redis_vision:6379') 
    parser.add_argument('-ts_u', '--ts_url', help='RedisTimesSeries URL', type=str, default='redis://redistimeseries_vision:6379')
    parser.add_argument('-skip', '--max_skipped_frame_allowed', help='Maximum skipped frame allowed', type=int, default=5)
    args = parser.parse_args()
    

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    # Set up Redistimeseries connection
    ts_url = urlparse(args.ts_url)
    ts_conn = redis.Redis(host=ts_url.hostname, port=ts_url.port)
    if not ts_conn.ping():
        raise Exception('Redistimeseries unavailable')


    gen(TrackertoTimeSeries(conn, ts_conn, args))

