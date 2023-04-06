from collections import defaultdict
import json


def drawtail(tracklets,image): 
    frame_coordinates = defaultdict(list)      
    for tracklet in tracklets:
        if (tracklet._object_id == "tracking_info"):
            boxes = json.loads(tracklet.__repr__())    
            final_tracking_data = boxes["boxes"]
            for each_object in final_tracking_data:
                for each_bbox in each_object:
                    objectid = each_bbox['objectId']
                    x, y, _ = each_bbox['object_bbox']
                    coordinates = (x,y)
                    frame_coordinates[objectid].append(coordinates)
        for key in frame_coordinates.keys():
            points = frame_coordinates[key]
            image.line(points, fill='white', width=2)

def midpointcalculator(x1,x2,y1,y2):
    midpoint = ((x1+x2)/2 , (y1+y2)/2)
    return list(midpoint)