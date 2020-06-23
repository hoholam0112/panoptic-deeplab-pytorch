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

# Custom libs
from transforms import build_transforms, PanopticTargetGenerator, SemanticTargetGenerator
from utils import rgb2id, id2rgb, decode_polygon, encode_polygon

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

_ROAD_CONDITION_THINGS_LIST = sorted(list(_class_name_to_id.values()))

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
            crop_size: Tuple, crop size. (height, width)
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
        self.training = training # If training=False, target is None.
        self.crop_h, self.crop_w = crop_size
        self.mirror = mirror
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size
        self.mean = mean
        self.std = std

        self.pad_value = tuple([int(v * 255) for v in self.mean])
        self.ignore_label = 0
        self.label_pad_value = (0, 0, 0)

        # Define instance variable
        self.rgb2id = rgb2id
        self.id2rgb = id2rgb

        self.image_files = [] # list of file names
        self.image_elements = [] # image element in xml file
        for d in sorted(os.listdir(self.root)):
            # collect image files
            files = sorted(glob(os.path.join(self.root, d, '*.jpg')))
            self.image_files += files
            if self.training:
                # read xml files and collect image elements
                xml_path = glob(os.path.join(self.root, d, '*.xml'))[0]
                root_element = elemTree.parse(xml_path)
                image_elements = root_element.findall('./image')
                self.image_elements += image_elements

        # Define transforms
        self.transform = build_transforms(self, training)
        if self.training:
            if semantic_only:
                self.target_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)
            else:
                self.target_transform = PanopticTargetGenerator(
                    self.ignore_label, self.rgb2id, _ROAD_CONDITION_THINGS_LIST,
                    sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                    small_instance_area=small_instance_area,
                    small_instance_weight=small_instance_weight)
        else:
            self.target_transform = None


    def __getitem__(self, idx):
        # Read images and labels
        img_path = self.image_files[idx]
        assert os.path.exists(img_path), 'Cannot find image files: {}'.format(img_path)
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        img = np.array(img, dtype=np.uint8)
        # to access image file name when evaluating
        sample = {}
        sample['dataset_index'] = torch.as_tensor(idx, dtype=torch.long)
        sample['raw_size'] = torch.as_tensor(np.array([width, height]))

        if not self.training:
            img, _ = self.transform(img, None)
            sample['image'] = img
            return sample

        # Read label
        img_elem = self.image_elements[idx]
        print(img_path)
        print(img_elem.attrib['name'])
        segments = []
        label = np.zeros_like(img, dtype=np.uint8)
        instance_id = 12345
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
            cv2.fillPoly(label, [points], rgb) # assing RGB value to label 
            instance_id += 1
            # Append seg to segments
            seg['iscrowd'] = False
            segments.append(seg)

        img, label = self.transform(img, label)
        sample['image'] = img
        # Generate training target.
        if self.target_transform is not None:
            label_dict = self.target_transform(label, segments)
            sample.update(label_dict)
        return sample

    def __len__(self):
        return len(self.image_files)

