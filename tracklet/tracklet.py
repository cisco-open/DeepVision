# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json


class Tracklet():
    def __init__(self, object_id, object_class):
        self._object_id = object_id
        self._boxes = []  # one object (all the boxes)
        self._object_class = object_class  # "PERSON"
        self.skipped_frames = 0  # count
        

    def add_box(self, box):
        self._boxes.append(box)

    def increase_skip(self):
        self._skipped_frames += 1
        return self
    
    def reset_skipped_frames(self):
        self._skipped_frames = 0


    @property
    def skipped_frames(self):
        return self._skipped_frames
  
    
    @skipped_frames.setter
    def skipped_frames(self, skipped_frames):
        if skipped_frames < 0:
            raise ValueError('Number should be positive')
        else:
            self._skipped_frames = skipped_frames
    

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
        temp_json = {"object_id": self._object_id, "boxes": self._boxes, "skipped_frames": str(self.skipped_frames)}
        return json.dumps(temp_json, indent=4)


