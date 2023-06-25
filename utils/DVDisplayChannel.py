from utils.RedisStreamManager import RedisStreamManager
from typing import List
import json
from PIL import Image, ImageDraw, ImageFont

class DVMessage:
    def __init__(self, message_text: str, text_position: dict = None,
                 bounding_box: dict = None, color: str = "black", line_width: int = None, 
                 font_size: int = 20, font_color: str = "rgb(0, 0, 0)"):
        self.message_text = message_text
        self.text_position = text_position
        self.bounding_box = bounding_box
        self.color = color
        self.line_width = line_width
        self.font_size = font_size
        self.font_color = font_color

    def to_dict(self):
        return self.__dict__

class DVDisplayChannel(RedisStreamManager):
    def __init__(self, stream_name: str):
        super().__init__(stream_name)

    def write_message(self, dv_messages: List[DVMessage]) -> None:
        # Convert list of DVMessage objects to a list of dicts, then to a JSON string
        messages_json = json.dumps([dv_message.to_dict() for dv_message in dv_messages])
        super().write_message({b'message': messages_json})

    def read_message(self) -> List[DVMessage]:
        message = self._read_latest_message()
        if message is not None:
            messages_json = message[b'message'].decode('utf-8')
            messages_dict = json.loads(messages_json)
            return [DVMessage(**message_dict) for message_dict in messages_dict]
        return None

    def render_messages(self, image: Image, dv_messages: List[DVMessage]) -> Image:
        draw = ImageDraw.Draw(image)

        if dv_messages is None or len(dv_messages) == 0:
            return None


        for msg in dv_messages:
            position = msg.text_position  # assuming this is a dict with 'x' and 'y' keys
            #current fonts in /usr/share/fonts/truetype/dejavu
            #DejaVuSans-Bold.ttf  DejaVuSansMono-Bold.ttf  DejaVuSerif-Bold.ttf
            #DejaVuSans.ttf	     DejaVuSansMono.ttf       DejaVuSerif.ttf
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", msg.font_size)

            # You could add more properties to DVMessage such as text color, font, etc.
            draw.text((position['x'], position['y']), msg.message_text, font=font, fill=msg.font_color)  # default to black color
