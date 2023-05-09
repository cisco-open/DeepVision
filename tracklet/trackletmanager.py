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

from tracklet.tracklet import Tracklet

class TrackletManager:
    
    STATUS_ACTIVE = "active"
    STATUS_INACTIVE = "inactive"


    def __init__(self, max_skipped_frame_allowed: int = 5, tracklet_length = 50):
        """ object_tracker (dict) : tracket class objects
            max_skipped_frame_allowed (integer): max number of frames skipped
            ts_status_labels (dict) : object's label status 
        """
        
        self.object_tracker = {}
        self.max_skipped_frame_allowed = max_skipped_frame_allowed
        self.ts_status_labels = {}
        self.counter = 0
        self.tracklet_length = tracklet_length
        self.tracklets = []


    def group_bboxes_for_an_object(self, objects):
        for objectId, object_bbox in list(objects.items()):
            if self.object_tracker.get(objectId) == None:
                a_tracklet = Tracklet(objectId, "PERSON")
                a_tracklet.add_box(object_bbox)
                self.object_tracker[objectId] = a_tracklet
                self.ts_status_labels[objectId] = self.STATUS_ACTIVE                
            else:
                self.object_tracker[objectId].add_box(object_bbox)
                

    def detect_skipped_frames(self, objects):
        for k, v in list(self.object_tracker.items()):
            if objects.get(k) == None and v.skipped_frames != self.max_skipped_frame_allowed:  # still active
                self.object_tracker[k] = v.increase_skip()
                # add fake box
                self.object_tracker[k].add_box([-1, -1, -1, -1, -1])
                self.ts_status_labels[k] = ""  # pass
            else:
                v.reset_skipped_frames()
            # if object is at max limit, delete and label inactive.
            if v.skipped_frames == self.max_skipped_frame_allowed:
                self.ts_status_labels[k] = self.STATUS_INACTIVE
                del self.object_tracker[k]


    def tracklet_collection_for_tail_visualization(self, objects):
        if (self.counter%self.tracklet_length == 0):    
            self.object_tracker.clear()
        for objectId, midpoint in objects:
            if self.object_tracker.get(objectId) is None:
                a_tracklet = Tracklet(objectId, "PERSON")
                a_tracklet.add_box(midpoint)
                self.object_tracker[objectId] = a_tracklet
            else:
                self.object_tracker[objectId].add_box(midpoint)
        self.counter += 1


    def process_objects(self, objects):
        """
        Args:
            objects (dict): dict of objects in one frame
        """
        self.group_bboxes_for_an_object(objects)
        self.detect_skipped_frames(objects)

        return self.ts_status_labels
    
    def values(self):
        return self.object_tracker.values()
        
