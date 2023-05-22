import cv2


class VideoStream:
    def __init__(self, isfile=0, fps=0.0):
        self.isFile = not str(isfile).isdecimal()
        self.isfile = isfile
        self.cam = cv2.VideoCapture(self.isfile)
        if not self.isFile:

            self.cam.set(cv2.CAP_PROP_FPS, fps)
            self.fps = fps

            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        else:
            self.fps = self.cam.get(cv2.CAP_PROP_FPS)
            if self.fps != fps:
                raise Exception(f"The actual fps {self.fps} is different from the input fps {fps}")

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        ret_val, img0 = self.cam.read()
        if not ret_val and self.isFile:
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_val, img0 = self.cam.read()
        assert ret_val, 'Video Error'

        img = img0
        if not self.isFile:
            img = cv2.flip(img, 1)

        return self.count, img

    def video_sample_rate(self, target_fps):
        return round(self.fps / target_fps)

    def release_camera(self):
        return self.cam.release()