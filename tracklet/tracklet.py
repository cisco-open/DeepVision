import json


class Tracklet():
    def __init__(self, object_id, object_class):
        self._object_id = object_id
        self._boxes = []  # one object (all the boxes)
        self._object_class = object_class  # "PERSON"
        self._skipped_frames = 0  # count
        

    def add_box(self, box):
        self._boxes.append(box)

    def increase_skip(self):
        self._skipped_frames += 1
        return self

    @property
    def skipped_frames(self):
        return self._skipped_frames

    @property
    def objectId(self):
        return self._object_id

    @property
    def object_class(self):
        return self._object_class

    @property
    def object_bboxes(self):
        return self._boxes

    def __repr__(self):
        temp_json = {"object_id": self._object_id, "boxes": self._boxes, "skipped_frames": str(self._skipped_frames)}
        return json.dumps(temp_json, indent=4)


