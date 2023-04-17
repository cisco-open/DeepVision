import unittest

from tracklet.trackletmanager import TrackletManager


class TestTrackletManager(unittest.TestCase):
    """
    This class tests the trackletmanager functionalities.
    """
    def test_process_objects_active(self):
        obj = TrackletManager(3)
        res = obj.process_objects({'a': 1, 'b': 2})
        self.assertEqual(res, {'a': 'active', 'b': 'active'})

    def test_process_objects_inactive(self):
        obj = TrackletManager(1)
        res = obj.process_objects({'a': None})
        self.assertEqual(res, {'a': 'inactive'})
