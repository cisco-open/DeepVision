import argparse
import cv2
import logging
import redis
import time
import pickle
from urllib.parse import urlparse



class Video:
    def __init__(self, infile=0, fps=0.0):
        self.isFile = not str(infile).isdecimal()
        self.ts = time.time()
        self.infile = infile
        self.cam = cv2.VideoCapture(self.infile)
        if not self.isFile:

            # Webcam fps settings
            self.cam.set(cv2.CAP_PROP_FPS, fps)
            self.fps = fps
            
            # TODO: some cameras don't respect the fps directive
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        else:
            # Read input Video file fps
            self.fps = self.cam.get(cv2.CAP_PROP_FPS)
            if self.fps != fps:
                raise Exception(f"The actual fps {self.fps} is different from the input fps {fps}")
            
            
    # For Video file self.fps is input file fps, target_fps(--fps) is passed as argument. 
    def video_sample_rate(self, target_fps):
        return round(self.fps/target_fps)
        

    def cam_release(self):
        return self.cam.release()
    
    
    def __iter__(self):
        self.count = -1
        return self


    def __next__(self):
        self.count += 1

        # Read image
        ret_val, img0 = self.cam.read()
        if not ret_val and self.isFile:
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_val, img0 = self.cam.read()
        assert ret_val, 'Video Error'

        # Preprocess
        img = img0
        if not self.isFile:
            img = cv2.flip(img, 1)

        return self.count, img

    def __len__(self):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Input file (leave empty to use webcam)', nargs='?', type=str, default=None)
    parser.add_argument('-o', '--output', help='Output stream key name', type=str, default='camera:0')
    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('-w', '--webcam', help='Webcam device number', type=int, default=0)
    parser.add_argument('-v', '--verbose', help='Verbose output', type=bool, default=False)
    parser.add_argument('--count', help='Count of frames to capture', type=int, default=None)
    parser.add_argument('--fmt', help='Frame storage format', type=str, default='.jpg')
    parser.add_argument('--inputFps', help='Frames per second (webcam)', type=float, default=30.0)
    parser.add_argument('--outputFps', help='Frames per second (webcam)', type=float, default=10.0)
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=3000)
    args = parser.parse_args()

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port, health_check_interval=25)
    if not conn.ping():
        raise Exception('Redis unavailable')

    # Choose video source
    if args.infile is None:
        loader = Video(infile=args.webcam, fps=args.outputFps)  # Default to webcam
        # Treat different - args.fps (no need to use video_sample_rate)
        for (count, img) in loader:
             msg = {
                'frameId': count,
                'image': pickle.dumps(img)
            }
             _id = conn.xadd(args.output, msg, maxlen=args.maxlen)
        
        loader.cam_release()

    else:
        loader = Video(infile=args.infile, fps=args.inputFps)  # Unless an input file (image or video) was specified
        frame_id = 0 # start new frame count
        rate = loader.video_sample_rate(args.outputFps)
        for (count, img) in loader:
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
            if args.count is not None and count + 1 == args.count:
                logging.info('Stopping after {} frames.'.format(count))
                break
                
if __name__ == '__main__':
    main()