import redis
import json
from argparse import ArgumentParser
from urllib.parse import urlparse
from Ethosight import Ethosight
from utils.RedisStreamXreaderWriter import RedisStreamXreaderWriter
from utils.Utility import get_frame_data


class EthosightService:
    def __init__(self, xreader_writer: RedisStreamXreaderWriter):
        self.ethosight = Ethosight()
        self.xreader_writer = xreader_writer
        self.sample_size = sample_size
        self.embeddings = self.ethosight.load_embeddings_from_disk("Ethosight/general.embeddings")

    def affinity_scores(self):
        last_id, data = self.xreader_writer.xread_latest_available_message()
        while True:
            ref_id, data = self.xreader_writer.xread_by_id(last_id)
            aff_scores = self.ethosight.compute_affinity_scores(self.embeddings, get_frame_data(data))
            print(aff_scores, flush=True)
            self.xreader_writer.write_message({'affinity_scores', json.dumps(aff_scores)}, last_id)
            last_id = ref_id


def main():
    parser.add_argument('--input_stream', help='input stream key for coming frames', type=str, default="camera:0")
    parser.add_argument('--output_stream', help='output stream key for affinity scores', type=str,
                        default="camera:0:affscores")
    parser.add_argument('--sampleSize', type=int, default=1, help='frames sample size')
    parser.add_argument('--redis', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=3000)

    args = parser.parse_args()

    url = urlparse(args.redis)
    conn = Redis(host=url.hostname, port=url.port, health_check_interval=25)

    xreader_writer = RedisStreamXreaderWriter(args.input_stream, args.output_stream, conn, args.maxlen)

    ethosight_service = EthosightService(xreader_writer)
    ethosight_service.affinity_scores()


if __name__ == "__main__":
    main()
