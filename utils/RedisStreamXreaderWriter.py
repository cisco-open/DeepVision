from utils.RedisStreamManager import RedisStreamManager

import redis

class RedisStreamXreaderWriter(RedisStreamManager):

    def __init__(self, rstream_name, wstream_name, redis_connection, max_len):
        self.rstream_name = rstream_name
        self.wstream_name = wstream_name
        self.redis_conn = redis_connection
        self.max_len = max_len

    def xread_latest_available_message(self) -> tuple:
        try:
            response = self.redis_conn.ping()
            if response == True:
                print("Redis connection successful!")
            else:
                print("Unable to connect to Redis.")
        except redis.exceptions.ConnectionError:
            print("Error connecting to Redis.")
        stream_exists = self.redis_conn.exists(self.rstream_name)
        stream_key = self.rstream_name
        if stream_exists:
            print(f"The stream '{stream_key}' exists.")
        else:
            print(f"The stream '{stream_key}' does not exist.")
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
            self.redis_conn.xadd(self.wstream_name, message, maxlen=self.max_len)
        else:
            self.redis_conn.xadd(self.wstream_name, message, id=item_id, maxlen=self.max_len)
