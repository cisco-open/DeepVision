from utils.RedisStreamManager import RedisStreamManager
import json

class DVDisplayChannel(RedisStreamManager):
    def __init__(self, stream_name: str):
        super().__init__(stream_name)

    def write_message(self, message_text: str, text_position: dict = None,
                      bounding_box: dict = None, color: str = None, line_width: int = None, 
                      font_size: int = None, font_color: str = None) -> None:
        message = {
            "message_text": message_text,
            "text_position": text_position,
            "bounding_box": bounding_box,
            "color": color,
            "line_width": line_width,
            "font_size": font_size,
            "font_color": font_color
        }

        # Filter out None values
        message = {k: v for k, v in message.items() if v is not None}
        # Convert the whole message to a JSON string
        message_json = json.dumps(message)

        super().write_message({b'message': message_json})

    def read_message(self) -> dict:
        message = self._read_latest_message()

        if message is not None:
            message_json = message[b'message'].decode('utf-8')
            message = json.loads(message_json)

        return message
