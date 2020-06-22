import os
import numpy as np
import torch
from PIL import Image
from glob import glob
import xml.etree.ElementTree as elemTree
import random
import cv2
from torchvision.transforms import functional as F
import sys
# sys.setrecursionlimit(10000)

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
    """Converts the color to panoptic label.
    Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
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
    """ Encode instance id as RGB value """
    assert isinstance(id, int)

# 21 classes. background = 0
_class_name_to_id = {'sidewalk_blocks' : 1, 'alley_damaged' : 2, 'sidewalk_damaged' : 3, 
             'caution_zone_manhole': 4, 'braille_guide_blocks_damaged':5, 'alley_speed_bump':6,
             'roadway_crosswalk':7,'sidewalk_urethane':8, 'caution_zone_repair_zone':9,
             'sidewalk_asphalt':10, 'sidewalk_other':11, 'alley_crosswalk':12,
             'caution_zone_tree_zone':13, 'caution_zone_grating':14, 'roadway_normal':15,
             'bike_lane':16, 'caution_zone_stairs':17, 'alley_normal':18, 
             'sidewalk_cement':19,'braille_guide_blocks_normal':20, 'sidewalk_soil_stone': 21}

_class_id_to_name = {v:k for k,v in _class_name_to_id.items()}

def class_name_to_id(class_name):
    return _class_name_to_id[class_name]

def class_id_to_name(class_id):
    return _class_id_to_name[class_id]

class BaseDataset(object):
    def __init__(self,
                 root,
                 training=True,
                 crop_size=(513, 1025),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 semantic_only=False,
                 ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1):
        """Cityscapes panoptic segmentation dataset.
        Arguments:
            root: Str, root directory.
            training: Bool, for training or testing.
            crop_size: Tuple, crop size.
            mirror: Bool, whether to apply random horizontal flip.
            min_scale: Float, min scale in scale augmentation.
            max_scale: Float, max scale in scale augmentation.
            scale_step_size: Float, step size to select random scale.
            mean: Tuple, image mean.
            std: Tuple, image std.
            semantic_only: Bool, only use semantic segmentation label.
            ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
            small_instance_area: Integer, indicates largest area for small instances.
            small_instance_weight: Integer, indicates semantic loss weights for small instances.
        """
        self.root = root
        self.transform = transform
        self.training = training # If training=False, target is None.
        self.image_files = [] # list of file names
        self.image_elements = [] # image element in xml file
        
        for d in sorted(os.listdir(self.root)):
            # collect image files
            files = sorted(glob(os.path.join(self.root, d, '*.jpg')))
            self.image_files += files
            if training:
                # read xml files and collect image elements
                xml_path = glob(os.path.join(self.root, d, '*.xml'))[0]
                root_element = elemTree.parse(xml_path)
                image_elements = root_element.findall('./image')
                self.image_elements += image_elements
        if training:
            self.target_transform = PanopticTargetGenerator(
                self.ignore_label, self.rgb2id, _CITYSCAPES_THING_LIST,
                sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                small_instance_area=small_instance_area,
                small_instance_weight=small_instance_weight)
        else:
            self.target_transform = None
        
    def __getitem__(self, idx):
        # Read images and labels
        img_path = self.image_files[idx]
        img = Image.open(img_path)
        width, height = img.size
        # to access image file name when evaluating
        target = {}
        target['dataset_index'] = torch.as_tensor(idx, dtype=torch.long)
        
        if not self.training:
            return img, target
        
        img_elem = self.image_elements[idx]
        
        print(img_path)
        print(img_elem.attrib['name'])
        
        segments = []
        panoptic = np.zeros_like(img, dtype=np.int32)
        instance_id = 0
        for polygon_elem in img_elem.findall('./polygon'):
            seg = {}
            # Read and process class_name
            class_name = polygon_elem.attrib['label']
            if class_name != 'bike_lane': # class 'bike_lane' has no attribute
                if len(polygon_elem.findall('attribute')) == 0: # Exception handling
                    continue
                class_name += '_'+polygon_elem.findall('attribute')[0].text
            seg['category_id'] = class_name_to_id(class_name)
            # Read and decode polygon string to np.array
            polygon_str = polygon_elem.attrib['points']
            points = decode_polygon(polygon_str)
            # Bounding box
            x1 = np.min(points[:, 0])
            y1 = np.min(points[:, 1])
            x2 = np.max(points[:, 0])
            y2 = np.max([points[:, 1]])
            seg['bbox'] = [x1, y1, x2, y2]
            # Encode instance id
            seg['id'] = instance_id
            rgb = id2rgb(instance_id) # encode instance id to RGB value 
            cv2.fillPoly(panoptic, [points], rgb) # assing RGB value to panoptic
            instance_id += 1
            # Append seg to segments
            seg['iscrowd'] = False
            segments.append(seg)
                
            img, panoptic = self.transform(img, panoptic)
        
        
        
        return img, target

    def __len__(self):
        return len(self.imgs)

    
    
class Compose(object):
    def __init__(self, transforms):
        """ list of transform operations """
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
        transforms.append(T.RandomHorizontalFlip(0.3))
    return T.Compose(transforms)
                   
def collate_fn(batch):
    return tuple(zip(*batch))
                   
def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
                   
def collate_fn(batch):
    return tuple(zip(*batch))

def make_dataset(root):
    folder_list = glob(root+'/*')
    dataset = BaseDataset(folder_list[0], get_transform(train=False))
    for fpath in folder_list[1:]:
        dataset_test = BaseDataset(fpath, get_transform(train=False))
        dataset = torch.utils.data.ConcatDataset([dataset, dataset_test])
    return dataset

train_dir = './dataset/train'

for d in sorted(os.listdir(train_dir)):
    files = sorted(glob(os.path.join(train_dir, d, '*.jpg')))
#     print(files)
    print(get_image_id(files[0]))
    break
    #     self.image_files.append(files)

