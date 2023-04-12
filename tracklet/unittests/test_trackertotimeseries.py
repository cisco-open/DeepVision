from unittest import mock

redis_mock = mock.MagicMock()
import sys

sys.modules['redistimeseries.client'] = redis_mock
from tracklet import trackertotimeseries
import unittest


class TestTrackerToTimeSeries(unittest.TestCase):

    def test_process_objects_active(self):
        conn_mock = mock.Mock()
        arg_mock = mock.MagicMock()
        arg_mock.camera = 'camera:0'
        arg_mock.max_skipped_frame_allowed = 3
        obj = trackertotimeseries.TrackertoTimeSeries(conn_mock, redis_mock,arg_mock)
        res = obj.process_statuses({'a': 'active'})
        self.assertEqual(res, None)

    def test_process_objects_inactive(self):
        conn_mock = mock.Mock()
        arg_mock = mock.MagicMock()
        arg_mock.camera = 'camera:0'
        arg_mock.max_skipped_frame_allowed = 3
        obj = trackertotimeseries.TrackertoTimeSeries(conn_mock, redis_mock,arg_mock)
        res = obj.process_statuses({'a': 'inactive'})
        self.assertEqual(res, None)

    def test_process_objects_none(self):
        conn_mock = mock.Mock()
        arg_mock = mock.MagicMock()
        arg_mock.camera = 'camera:0'
        arg_mock.max_skipped_frame_allowed = 3
        obj = trackertotimeseries.TrackertoTimeSeries(conn_mock, redis_mock,arg_mock)
        res = obj.process_statuses({'a': ''})
        self.assertEqual(res, None)

    def test_get_last(self):
        conn_mock = mock.MagicMock()
        pipeline_mock = mock.MagicMock()
        conn_mock.pipeline.return_value = pipeline_mock
        pipeline_mock.execute.return_value = [[(b'1679605954574-0', {b'refId': b'1679605954464-0',
                                                                     b'tracking': b'{"frameId": 48123, "tracking_info": [{"objectId": 29496, "object_bbox": [792.0845947265625, 309.7529296875, 851.9524536132812, 458.04107666015625, 0.9981083869934082], "class": "PERSON"}, {"objectId": 30245, "object_bbox": [954.7723388671875, 244.40615844726562, 1007.9391479492188, 398.2245178222656, 0.988837718963623], "class": "PERSON"}, {"objectId": 30047, "object_bbox": [98.80311584472656, 381.30316162109375, 157.2754364013672, 544.6788940429688, 0.9813268780708313], "class": "PERSON"}, {"objectId": 29956, "object_bbox": [533.0682373046875, 247.82191467285156, 616.8584594726562, 447.8920593261719, 0.9595788717269897], "class": "PERSON"}]}'})]]

        arg_mock = mock.MagicMock()
        arg_mock.camera = 'camera:0'
        arg_mock.max_skipped_frame_allowed = 3

        obj = trackertotimeseries.TrackertoTimeSeries(conn_mock, redis_mock,arg_mock)
        res = obj.get_last()
        self.assertEqual(res, True)
