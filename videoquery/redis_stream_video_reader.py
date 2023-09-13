from typing import Dict
import pickle
from vqpy.backend.operator import CustomizedVideoReader
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter


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
