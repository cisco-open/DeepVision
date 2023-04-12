import json

import unittest
from tracklet.tracklet import Tracklet

class TestTracklet(unittest.TestCase):

    def test_add_box(self):

        obj = Tracklet('', '')
        obj.add_box(-1)
        obj.add_box(-1)
        obj.add_box(-1)
        obj.add_box(-1)
        obj.add_box(-1)
        self.assertEqual(obj._boxes, [-1, -1, -1, -1, -1])

    def test_increase_skip(self):
        from tracklet.tracklet import Tracklet
        obj = Tracklet('', '')
        obj.increase_skip()
        self.assertEqual(obj._skipped_frames, 1)

    def test_skipped_frames(self):
        from tracklet.tracklet import Tracklet
        obj = Tracklet('', '')
        self.assertEqual(obj.skipped_frames, 0)

    def test_objectId(self):
        from tracklet.tracklet import Tracklet
        obj = Tracklet('', '')
        self.assertEqual(obj.objectId, '')

    def test_object_class(self):
        from tracklet.tracklet import Tracklet
        obj = Tracklet('', '')
        self.assertEqual(obj.object_class, '')

    def test__repr__(self):
        from tracklet.tracklet import Tracklet
        obj = Tracklet('', '')
        res = obj.__repr__()
        self.assertEqual(json.loads(res), {
            "object_id": "",
            "boxes": [],
            "skipped_frames": "0"
        })


