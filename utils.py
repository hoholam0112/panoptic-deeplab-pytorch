import numpy as np

def get_image_id(file_path):
    """ extract image id from file path """
    return file_path.split('/')[-1].split('.')[0]

def decode_polygon(polygon_str):
    """ decode polygon string representation as np.array """
    point_strs = polygon_str.split(';')
    seq = []
    for p in point_strs:
        x , y = p.split(',')
        seq.append([int(float(x)), int(float(y))])
    return np.array(seq)

def encode_polygon(points):
    """ encode np.array points constructing polygon as a string """
    pass

def rgb2id(color):
    """Converts the color encoding 256-base number to panoptic label.
    Args:
        color: Ndarray or a tuple, color encoded image.
    Returns:
        Panoptic label.
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

def id2rgb(id):
    """ Encode instance id as RGB value. 256-base number """
    assert isinstance(id, int)
    if id > 256**3 - 1:
        raise ValueError('''Given id excess range.
                {} is larger than (256**3 - 1)'''.format(id))
    b = id // 256 // 256
    id -= b * 256 * 256
    g = id // 256
    id -= g * 256
    r = id % 256
    return (r, g, b)

