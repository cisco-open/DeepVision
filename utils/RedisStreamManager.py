import redis
import json

class RedisStreamManager:
    def __init__(self, stream_name: str):
        self.stream_name = stream_name
        self.redis_conn = redis.Redis(host='redis', port=6379, db=0)
        self.clear_stream()

    def write_message(self, message: dict) -> None:
        self.redis_conn.xadd(self.stream_name, message)

    def _read_latest_message(self) -> dict:
        messages = self.redis_conn.xrevrange(self.stream_name, count=1)
        if messages:
            _, message = messages[0]
            return message
        else:
            return None

    def clear_stream(self) -> None:
        self.redis_conn.delete(self.stream_name)