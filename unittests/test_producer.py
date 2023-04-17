import sys
import unittest
from unittest import mock

arg_mock = mock.MagicMock()
redis_mock = mock.MagicMock()
sys.modules['redis'] = redis_mock
sys.modules['argparse'] = arg_mock

import producer


class TestProducer(unittest.TestCase):
    """
    This class tests the producer functionalities.
    """
    def test_main_with_infile(self):
        redis_mock2 = mock.MagicMock()
        redis_mock.Redis.return_value = redis_mock2
        redis_mock2.ping.return_value = True
        arg_mock2 = mock.MagicMock()
        arg_mock.ArgumentParser.return_value = arg_mock2
        arg_mock3 = mock.MagicMock()
        arg_mock2.parse_args.return_value = arg_mock3
        arg_mock3.url = 'redis://127.0.0.1:6379'
        arg_mock3.infile = 'data/race.mp4'
        arg_mock3.output = 'camera:0'
        arg_mock3.webcam = 0
        arg_mock3.verbose = False
        arg_mock3.count = 5
        arg_mock3.fmt = '.jpg'
        arg_mock3.inputFps = 30.0
        arg_mock3.maxlen = 3000
        arg_mock3.outputFps = 10.0
        res = producer.main()
        self.assertIsNone(res)

    def test_main_exception(self):
        redis_mock2 = mock.MagicMock()
        redis_mock.Redis.return_value = redis_mock2
        redis_mock2.ping.return_value = False
        arg_mock2 = mock.MagicMock()
        arg_mock.ArgumentParser.return_value = arg_mock2
        arg_mock3 = mock.MagicMock()
        arg_mock2.parse_args.return_value = arg_mock3
        arg_mock3.url = 'redis://127.0.0.1:6379'
        arg_mock3.infile = 'data/race.mp4'
        arg_mock3.output = 'camera:0'
        arg_mock3.webcam = 0
        arg_mock3.verbose = False
        arg_mock3.count = 5
        arg_mock3.fmt = '.jpg'
        arg_mock3.inputFps = 30.0
        arg_mock3.maxlen = 3000
        arg_mock3.outputFps = 10.0
        with self.assertRaises(Exception):
            producer.main()

    def test_video_sample_rate(self):
        obj = producer.Video(0, 30)
        res = obj.video_sample_rate(30)
        self.assertEqual(res, 1)

    def test_cam_release(self):
        obj = producer.Video(0, 30)
        res = obj.cam_release()
        self.assertIsNone(res)

    def test___iter__(self):
        obj = producer.Video(0, 30)
        res = obj.__iter__()
        self.assertEqual(res.count, -1)
        res1 = obj.__len__()
        self.assertEqual(res1, 0)
        obj.isFile = True
