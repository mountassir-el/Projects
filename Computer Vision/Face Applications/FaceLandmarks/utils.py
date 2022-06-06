import numpy as np

def rect_to_bbox(rect):
    """Transform a rectangle into a bounding box.

    :param rect: the rectangle
    :type rect: dlib.rectangle
    :return: bounding box
    :rtype: list
    """
    xmin, ymin = rect.left(), rect.top()
    xmax, ymax = rect.right(), rect.bottom()

    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def shape_to_np(shape, dtype = "int"):
    """Transform shape into numpy array.

    :param shape: the shape.
    :type shape: dlib.shape
    :param dtype: the numpy array dtype, defaults to "int"
    :type dtype: str, optional
    :return: the numpy array corresponding to the shape
    :rtype: numpy.ndarray
    """
    ret = np.zeros((68,2), dtype = dtype)
    for i in range(68):
        x, y = shape.part(i).x, shape.part(i).y
        ret[i] = (x, y)
    return ret
