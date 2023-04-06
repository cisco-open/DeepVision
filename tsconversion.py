# Different algos to convert tracker tracking info to timeseries


def aspect_ratio(tl_x, tl_y, br_x, br_y):
    width = br_x - tl_x
    height = br_y - tl_y

    return round(height / width)


def perspective_invariant():
    pass
