
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
