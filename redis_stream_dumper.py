import json
import os
import time
from redis import Redis
from argparse import ArgumentParser
from urllib.parse import urlparse
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from utils.Utility import clean_stream

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--input_stream', help='input stream for dumping', type=str, default="camera:0:affscores")
    parser.add_argument('--input_stream_original', help='input stream for initial frames', type=str, default="camera:0")
    parser.add_argument('--filePath', help='file where to dump', type=str)
    parser.add_argument('--batchSize', type=int, help='batch size', default=50)

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)

    xreader_writer = RedisStreamXreaderWriter(args.input_stream, "", conn, 0)

    last_id = '0'
    was_data = False
    retry_count = 0
    with open(args.filePath, 'w') as file:
        while True:
            ref_id, data = xreader_writer.xread_by_id(last_id)
            if ref_id:
                retry_count = 0
                was_data = True
                message_json = data[b'message'].decode('utf-8')
                file.write(message_json + '\n')
                last_id = ref_id    
            elif was_data:
                if retry_count == 3:
                    clean_stream(conn, args.input_stream_original)
                    clean_stream(conn, args.input_stream)
                    break
                
                time.sleep(3)
                retry_count = retry_count + 1





