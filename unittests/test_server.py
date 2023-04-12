import pickle
import unittest

from unittest import mock
import sys


import numpy as np
from PIL import Image

from server import RedisImageStream

#
arg_mock = mock.MagicMock()
redis_mock = mock.MagicMock()
sys.modules['redis'] = redis_mock
sys.modules['argparse'] = arg_mock

import producer


class TestServer(unittest.TestCase):
    """ The function to test (would usually be loaded from a module outside this file).
    """
    def test_server(self):
        arg_mock = mock.MagicMock()
        arg_mock.camera = 'camera:0'
        arg_mock.boxes = 'camera:0:mot'
        arg_mock.field = 'image'
        arg_mock.fmt = '.jpg'
        arg_mock.url = 'redis://127.0.0.1:6379'
        conn_mock = mock.MagicMock()
        pipeline_mock = mock.MagicMock()
        conn_mock.pipeline.return_value = pipeline_mock
        pipeline_mock.execute.return_value = 1,[(b'1679605954574-0', {b'refId': b'1679605954464-0',
                                                                     b'tracking': b'{"frameId": 48123, "tracking_info": [{"objectId": 29496, "object_bbox": [792.0845947265625, 309.7529296875, 851.9524536132812, 458.04107666015625, 0.9981083869934082], "class": "PERSON"}, {"objectId": 30245, "object_bbox": [954.7723388671875, 244.40615844726562, 1007.9391479492188, 398.2245178222656, 0.988837718963623], "class": "PERSON"}, {"objectId": 30047, "object_bbox": [98.80311584472656, 381.30316162109375, 157.2754364013672, 544.6788940429688, 0.9813268780708313], "class": "PERSON"}, {"objectId": 29956, "object_bbox": [533.0682373046875, 247.82191467285156, 616.8584594726562, 447.8920593261719, 0.9595788717269897], "class": "PERSON"}]}'})]

        im = Image.open('unittests/data/sample.jpg')
        img = np.asarray(im)

        msg = {
            b'frameId': '1'.encode('utf-8'),
            b'image': pickle.dumps(img)
        }
        conn_mock.xread.return_value = [(1, [(1, msg)])]
        obj = RedisImageStream(conn_mock, arg_mock)
        res = obj.get_last()
        self.assertIsNotNone(res)

    def test_server_without_tracking_stream(self):
        arg_mock = mock.MagicMock()
        arg_mock.camera = 'camera:0'
        arg_mock.boxes = 'camera:0:mot'
        arg_mock.field = 'image'
        arg_mock.fmt = '.jpg'
        arg_mock.url = 'redis://127.0.0.1:6379'
        conn_mock = mock.MagicMock()
        pipeline_mock = mock.MagicMock()
        conn_mock.pipeline.return_value = pipeline_mock

        im = Image.open('unittests/data/sample.jpg')
        img = np.asarray(im)
        msg =  {
            b'frameId': '1'.encode('utf-8'),
            b'image': pickle.dumps(img)
        }
        pipeline_mock.execute.return_value = [(b'1679605954574-0', msg)],None
        conn_mock.xread.return_value = [(1, [(1, msg)])]
        obj = RedisImageStream(conn_mock, arg_mock)
        res = obj.get_last()
        self.assertIsNotNone(res)