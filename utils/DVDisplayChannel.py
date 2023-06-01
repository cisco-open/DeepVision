from utils.RedisStreamManager import RedisStreamManager
import json

def decode_dict(input_value):
    if isinstance(input_value, bytes):
        return input_value.decode('utf-8')

    if isinstance(input_value, (str, int, float)):
        return input_value

    if isinstance(input_value, dict):
        return {decode_dict(key): decode_dict(value) for key, value in input_value.items()}

    if isinstance(input_value, (list, tuple)):
        return type(input_value)(decode_dict(element) for element in input_value)

    # For any other types, return the value as is
    return input_value

class DVDisplayChannel(RedisStreamManager):
    def __init__(self, stream_name: str):
        super().__init__(stream_name)

    def write_message(self, message_text: str, text_position: dict = None,
                      bounding_box: dict = None, color: str = None, line_width: int = None, 
                      font_size: int = None, font_color: str = None) -> None:
        message = {
            "message_text": message_text,
            "text_position": json.dumps(text_position) if text_position else None,
            "bounding_box": json.dumps(bounding_box) if bounding_box else None,
            "color": color,
            "line_width": line_width,
            "font_size": font_size,
            "font_color": font_color
        }

        # Filter out None values
        message = {k: v for k, v in message.items() if v is not None}
        super().write_message(message)

    def read_message(self) -> dict:
        message = self._read_latest_message()
        if message is not None:
            message = decode_dict(message)
        return message

    def read_message2(self) -> dict:
        message = self.read_latest_message()

        if message is not None:
            for key in ['text_position', 'bounding_box']:
                if message.get(key):
                    message[key] = json.loads(message[key])

        return message
