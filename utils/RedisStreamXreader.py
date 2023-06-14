from utils.RedisStreamManager import RedisStreamManager


class RedisStreamXreader(RedisStreamManager):

    def __init__(self, stream_name, redis_connection):
        self.stream_name = stream_name
        self.redis_conn = redis_connection

    def xread_latest_available_message(self) -> tuple:
        resp = self.redis_conn.xread({self.stream_name: '$'}, count=None, block=0)
        if resp:
            key, messages = resp[0]
            ref_id, data = messages[0]
            return ref_id, data
        else:
            return None, None

    def xread_by_id(self, item_id: str) -> tuple:
        resp = self.redis_conn.xread({self.stream_name: item_id}, count=1)
        if resp:
            key, messages = resp[0]
            ref_id, data = messages[0]
            return ref_id, data
        else:
            return None, None
