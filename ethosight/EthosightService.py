import json
import os
from redis import Redis
from argparse import ArgumentParser
from urllib.parse import urlparse
from Ethosight import Ethosight
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from utils.Utility import get_frame_data, NpEncoder, convert_redis_entry_id_to_mls


class EthosightService:
    def __init__(self, xreader_writer: RedisStreamXreaderWriter, embeddings_file_name: str, benchmark: bool):
        self.xreader_writer = xreader_writer
        self.ethosight = Ethosight(reasoner='reasoner')
        self.benchmark = benchmark
        self.embeddings = self.ethosight.load_embeddings_from_disk(f"embeddings/{embeddings_file_name}")

    def affinity_scores(self):
        last_id = '0'
        while True:
            ref_id, data = self.xreader_writer.xread_by_id(last_id)
            if data:
                print("data available", flush=True)
                aff_scores = self.ethosight.compute_affinity_scores(self.embeddings, get_frame_data(data))
                frame_id = get_frame_data(data, 'frameId')
                timestamp = convert_redis_entry_id_to_mls(ref_id.decode())
                message_json = json.dumps({'frame_id': frame_id, 'data': {'timestamp': timestamp, 'affinity_scores': aff_scores}}, cls=NpEncoder)
                print(f'Message: \n {message_json}', flush=True)
                self.xreader_writer.write_message({'message': message_json})
                last_id = ref_id

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_stream', help='input stream key for coming frames', type=str, default="camera:0")
    parser.add_argument('--output_stream', help='output stream key for affinity scores', type=str,
                        default="camera:0:affscores")
    parser.add_argument('--sampleSize', type=int, default=1, help='frames sample size')
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--embeddings', help='Label embeddings', type=str, default='general.embeddings')
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=5000)
    parser.add_argument('--benchmark', help='Benchmark mode', type=bool, default=False)

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)

    xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream, conn, None)

    clean_stream(conn, args.output_stream)
    ethosight_service = EthosightService(xreader_writer, args.embeddings, args.benchmark)
    ethosight_service.affinity_scores()


if __name__ == "__main__":
    main()
