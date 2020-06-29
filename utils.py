import numpy as np
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

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

def points_to_str(points):
    """ encode points (np.array of shape [N, 2]) as a string """
    polygon_str = ''
    for i in range(points.shape[0]):
        polygon_str += str(points[i, 0]) + ',' + str(points[i, 1]) +';'
    return polygon_str

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

def reverse_transform(image_tensor, mean, std):
    """Reverse the normalization on image.
    Args:
        image_tensor: torch.Tensor, the normalized image tensor.
    Returns:
        image: numpy.array, the original image before normalization.
    """
    dtype = image_tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=image_tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=image_tensor.device)
    image_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    image = image_tensor.mul(255)\
                        .clamp(0, 255)\
                        .byte()\
                        .permute(1, 2, 0)\
                        .cpu().numpy()
    return image

def create_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0] # no label -> Black
    colormap[1] = [153, 204, 255] # Sideblock normal -> Dark red
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 255, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    colormap[19] = [128, 64, 128]
    colormap[20] = [128, 128, 0]
    colormap[21] = [170, 255, 195] # Mint
    return colormap

def color_segmap(sem_hard):
    """ colorize segmentation map np.array of shape (H, W) """
    colormap = create_label_colormap()
    h, w = sem_hard.shape
    semantic_color = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id in range(0, 22):
        semantic_color[sem_hard == cat_id] = colormap[cat_id]
    return semantic_color

def color_mask(cat_id, mask):
    """ Colorize segmentation mask """
    colormap = create_label_colormap()
    r, g, b = colormap[cat_id]
    mask_color = np.zeros([h, w, 3], dtype=np.uint8)
    mask_color[:,:,0][mask] = r
    mask_color[:,:,1][mask] = g
    mask_color[:,:,2][mask] = b
    return mask_color

class NoiseRemover():
    """ Remove noisy part (coarse borderlines or small ugly spots)
    in binary mask image by erosion and dilation """
    def __init__(self, kernel_size=7, erode_iter=1, dilate_iter=2):
        self.kernel_size = kernel_size
        self.erode_iter = erode_iter
        self.dilate_iter = dilate_iter
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, mask):
        kernel = np.ones((7,7),np.uint8)
        mask_erode = cv2.erode(
		mask.astype(np.uint8), self.kernel, iterations=self.erode_iter)
        mask_process = cv2.dilate(
		mask_erode, self.kernel, iterations=self.dilate_iter)
        mask_process = mask_process.astype(np.bool)
        return mask_process
