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

def draw_tail(tracklets, image, colors):
    for tracklet in tracklets:
        counter = 0
        radius = 4
        bbox = tracklet.object_bboxes
        color = colors[tracklet.objectId] if tracklet.objectId in colors else 'black'
        if color != 'black':
            image.line(bbox, fill=color, width=3)
        for box in bbox:
            if (counter % 10 == 0):
                circle_color = (32, 38, 46)
                if color != 'black':
                    image.ellipse((box[0]-radius, box[1]-radius, box[0]+radius, box[1]+radius),
                                  fill=circle_color, outline=circle_color)
            counter += 1


def midpoint_calculate(x1, x2, y1, y2):
    return ((x1+x2)/2,(y1+y2)/2)

def get_tracking_entry_with_midpoint(tracking_entry, midpoint):

    return (tracking_entry['objectId'], midpoint)

def update_midpoint_to_tracklets(x1,x2,y1,y2,tracking_entry):
    midpoint = midpoint_calculate(x1, x2, y1, y2)
    tracking_entry_with_midpoint = get_tracking_entry_with_midpoint(tracking_entry, midpoint)
    return tracking_entry_with_midpoint
