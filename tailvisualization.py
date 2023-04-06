from collections import defaultdict
import json


def drawlines(tracklets,image):       
    for value in tracklets:
        frame_coordinates = defaultdict(list)
        if (value._object_id == "tracking_info"):
            boxes = value.__repr__()    
            boxes = json.loads(boxes)
            final_data = boxes["boxes"]
            for data in final_data:
                for d in data:
                    frameid = d['objectId']
                    x, y, _ = d['object_bbox']
                    coordinates = (x,y)
                    frame_coordinates[frameid].append(coordinates)
                    for key in frame_coordinates.keys() :
                        points = frame_coordinates[key]
                        image.line(points, fill='white', width=2)

def midpointcalculator(x1,x2,y1,y2):
    midpoint = ((x1+x2)/2 , (y1+y2)/2)
    return list(midpoint)