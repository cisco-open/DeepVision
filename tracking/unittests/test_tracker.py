import argparse

import pickle
import unittest

from unittest import mock
import sys
from unittest.mock import patch

import numpy as np

arg_mock = mock.MagicMock()
redis_mock = mock.MagicMock()
sys.modules['redis'] = redis_mock
sys.modules['argparse'] = arg_mock
sys.modules['redis'] = redis_mock

sys.modules['mmtracking'] = arg_mock
sys.modules['mmtracking.mmtrack'] = arg_mock
sys.modules['mmtracking.mmtrack.apis'] = arg_mock

import tracking.tracker as tracker


class TestTracker(unittest.TestCase):
    """
    The function to test (would usually be loaded from a module outside this file).
    """

    @patch('tracking.tracker.results2outs')
    def test_main_with_no_infile(self, mock_results):
        ids_mock = mock.MagicMock()
        mock_results.return_value = ids_mock
        ids_mock.get.return_value = (1, 2, 3)
        redis_mock2 = mock.MagicMock()
        redis_mock.Redis.return_value = redis_mock2
        redis_mock2.ping.return_value = True
        redis_mock2.xadd.side_effect = [1, Exception]
        img = np.array([[[62, 62, 62], [61, 61, 61], [63, 63, 63]]])
        msg = {
            b'frameId': '1'.encode('utf-8'),
            b'image': pickle.dumps(img)
        }
        redis_mock2.xread.return_value = [(1, [(1, msg)])]
        arg_mock2 = mock.MagicMock()
        arg_mock.ArgumentParser.return_value = arg_mock2
        arg_mock2.parse_args.return_value = argparse.Namespace(input_stream='camera:0', classId='PERSON',
                                                               output_stream='camera:0:mot',
                                                               checkpoint='',
                                                               device='cuda:0',
                                                               redis='redis://127.0.0.1:6379',
                                                               maxlen=3000,
                                                               config='')
        with self.assertRaises(Exception):
            tracker.main()

    def test_results2outs_error(self):
        bbox = np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        with self.assertRaises(NotImplementedError):
            tracker.results2outs(bbox, bbox, bbox)

    def test_results2outs_5(self):
        bbox = np.ones((1, 5), dtype=np.float32), np.ones((1, 5), dtype=np.float32)
        actual_response = tracker.results2outs(bbox, bbox, bbox)
        self.assertEqual(actual_response['labels'].size, np.array([0, 1], dtype=np.int64).size)

    def test_results2outs_6(self):
        bbox = np.ones((1, 6), dtype=np.float32), np.ones((1, 6), dtype=np.float32)
        actual_response = tracker.results2outs(bbox, bbox, bbox)
        self.assertEqual(actual_response['labels'].size, np.array([0, 1], dtype=np.int64).size)

    def test_results2outs_masks(self):
        bbox = np.ones((1, 6), dtype=np.float32), np.ones((1, 6), dtype=np.float32)
        actual_response = tracker.results2outs(bbox, bbox, (np.zeros((0, 6), dtype=np.float32),np.zeros((0, 6), dtype=np.float32)))
        self.assertEqual(actual_response['labels'].size, np.array([0, 1], dtype=np.int64).size)

