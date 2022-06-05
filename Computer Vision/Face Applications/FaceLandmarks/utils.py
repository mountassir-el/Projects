import numpy as np

def rect_to_bbox(rect):
    xmin, ymin = rect.left(), rect.top()
    xmax, ymax = rect.right(), rect.bottom()

    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def shape_to_np(shape, dtype = "int"):
    ret = np.zeros((68,2), dtype = dtype)
    for i in range(68):
        x, y = shape.part(i).x, shape.part(i).y
        ret[i] = (x, y)
    return ret
