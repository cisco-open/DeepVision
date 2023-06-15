from utils.RedisStreamManager import RedisStreamManager


class RedisStreamXreaderWriter(RedisStreamManager):

    def __init__(self, rstream_name, wstream_name, redis_connection):
        self.rstream_name = rstream_name
        self.wstream_name = wstream_name
        self.redis_conn = redis_connection

    def xread_latest_available_message(self) -> tuple:
        resp = self.redis_conn.xread({self.rstream_name: '$'}, count=None, block=0)
        if resp:
            key, messages = resp[0]
            ref_id, data = messages[0]
            return ref_id, data
        else:
            return None, None

    def xread_by_id(self, item_id: str) -> tuple:
        resp = self.redis_conn.xread({self.rstream_name: item_id}, count=1)
        if resp:
            key, messages = resp[0]
            ref_id, data = messages[0]
            return ref_id, data
        else:
            return None, None
        
    def write_message(self, message: dict, item_id: str = None) -> None:
        if not item_id:
            self.redis_conn.xadd(self.wstream_name, message)
        else:
            self.redis_conn.xadd(self.wstream_name, message, id=item_id)
