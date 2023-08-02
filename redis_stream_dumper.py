import json
import os
from redis import Redis
from argparse import ArgumentParser
from urllib.parse import urlparse
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--input_stream', help='input stream for dumping', type=str, default="camera:0:affscores")
    parser.add_argument('--filePath', help='file where to dump', type=str)
    parser.add_argument('--batchSize', type=int, default=50, help='batch size')

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)

    xreader_writer = RedisStreamXreaderWriter(args.input_stream, "", conn, 0)

    last_id = '0'
    with open(args.filePath, 'w') as file:
        while True:
            ref_id, messages = xreader_writer.xread_by_id_batched(last_id, args.batchSize)
            if ref_id:
                for entry in messages:
                    messages_json = message[b'message'].decode('utf-8')
                    file.write(messages_json + '\n')
            else:
                break






