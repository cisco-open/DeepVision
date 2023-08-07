import json
import os
import threading
from redis import Redis
from argparse import ArgumentParser
from urllib.parse import urlparse
from EthosightInitializer import EthosightSingleton
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from utils.Utility import get_frame_data, NpEncoder, convert_redis_entry_id_to_mls


class EthosightService:
    def __init__(self, xreader_writer: RedisStreamXreaderWriter, embeddings_file_name: str):
        self.xreader_writer = xreader_writer
        self.ethosight = EthosightSingleton()
        self.embeddings = self.ethosight.load_embeddings_from_disk(f"embeddings/{embeddings_file_name}")

    def affinity_scores(self):
        was_data = False
        last_id = '0'
        while True:
            ref_id, data = self.xreader_writer.xread_by_id(last_id)
            if data:
                was_data = True
                aff_scores = self.ethosight.compute_affinity_scores(self.embeddings, get_frame_data(data))
                frame_id = get_frame_data(data, 'frameId')
                timestamp = convert_redis_entry_id_to_mls(ref_id.decode())
                message_json = json.dumps({'frame_id': frame_id, 'data': {'timestamp': timestamp, 'affinity_scores': aff_scores}}, cls=NpEncoder)
                self.xreader_writer.write_message({'message': message_json})
                last_id = ref_id
            elif was_data:
                break


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_stream', help='comma separated input streams list', type=str, default="camera:0")
    parser.add_argument('--output_stream', help='comma separated output streams for affinity scores', type=str,
                        default="camera:0:affscores")
    parser.add_argument('--sampleSize', type=int, default=1, help='frames sample size')
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--embeddings', help='Label embeddings', type=str, default='general.embeddings')
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=5000)

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)

    input_streams = args.input_stream.rstrip(',').split(',')
    output_streams = args.output_stream.rstrip(',').split(',')

    for input_stream, output_stream in zip(input_streams, output_streams):
        xreader_writer = RedisStreamXreaderWriter(input_stream, output_stream, conn, None)
        ethosight_service = EthosightService(xreader_writer, args.embeddings)
        thread = threading.Thread(target=ethosight_service.affinity_scores)
        thread.start()
        thread.join()


if __name__ == "__main__":
    main()
