from collections import defaultdict
import json


def draw_tail(tracklets, image):
    # frame_coordinates = defaultdict(list)
    # for tracklet in tracklets:
    #     if (tracklet._object_id == "tracking_info"):
    #         boxes = json.loads(tracklet.__repr__())
    #         final_tracking_data = boxes["boxes"]
    #         for each_object in final_tracking_data:
    #             for each_bbox in each_object:
    #                 objectid = each_bbox['objectId']
    #                 x, y, _ = each_bbox['object_bbox']
    #                 coordinates = (x,y)
    #                 frame_coordinates[objectid].append(coordinates)
    #     for key in frame_coordinates.keys():
    #         points = frame_coordinates[key]
    #         image.line(points, fill='white', width=2)

    for tracklet in tracklets:
        image.line(tracklet.object_bboxes, fill='white', width=2)


def midpoint_calculate(x1, x2, y1, y2):
    return ((x1+x2)/2,(y1+y2)/2)

def get_tracking_entry_with_midpoint(tracking_entry, midpoint):

    return (tracking_entry['objectId'], midpoint)
