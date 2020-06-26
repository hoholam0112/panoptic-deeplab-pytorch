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
from utils import rgb2id, id2rgb, decode_polygon

# 21 classes. background = 0
_class_name_to_id = {'sidewalk_blocks' : 1, 'alley_damaged' : 2, 'sidewalk_damaged' : 3,
             'caution_zone_manhole': 4, 'braille_guide_blocks_damaged':5, 'alley_speed_bump':6,
             'roadway_crosswalk':7,'sidewalk_urethane':8, 'caution_zone_repair_zone':9,
             'sidewalk_asphalt':10, 'sidewalk_other':11, 'alley_crosswalk':12,
             'caution_zone_tree_zone':13, 'caution_zone_grating':14, 'roadway_normal':15,
             'bike_lane':16, 'caution_zone_stairs':17, 'alley_normal':18,
             'sidewalk_cement':19,'braille_guide_blocks_normal':20, 'sidewalk_soil_stone': 21}

_class_id_to_name = {v:k for k,v in _class_name_to_id.items()}

_ROAD_CONDITION_THINGS_LIST = sorted(list(_class_name_to_id.values()))
_LABEL_DIVISOR = 10000

def class_name_to_id(class_name):
    return _class_name_to_id[class_name]

def class_id_to_name(class_id):
    return _class_id_to_name[class_id]

def get_thing_list():
    return _ROAD_CONDITION_THINGS_LIST

def get_label_divisor():
    return _LABEL_DIVISOR

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
        self.mask_dir = root + '_mask'
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
        for d in sorted(os.listdir(self.root)):
            # collect image files
            files = sorted(glob(os.path.join(self.root, d, '*.jpg')))
            self.image_files += files

        if self.training:
            if not os.path.isdir(self.mask_dir):
                raise FileNotFoundError('Mask directory is not found: {}'.format(self.mask_dir))
            self.mask_files = [] # mask file list 
            for d in sorted(os.listdir(self.mask_dir)):
                # collect image files
                files = sorted(glob(os.path.join(self.mask_dir, d, '*.png')))
                self.mask_files += files

            if len(self.image_files) != len(self.mask_files):
                raise RuntimeError('''number of mask files is different from that of image files.
                        Please excute mask_generate.py until finished.''')

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

        # read mask
        mask_path = self.mask_files[idx]
        assert os.path.exists(mask_path), 'Cannot find mask file: {}'.format(mask_path)
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask, dypte=np.unit8)

        img, label = self.transform(img, mask)
        sample['image'] = img
        # Generate training target.
        if self.target_transform is not None:
            label_dict = self.target_transform(label, _LABEL_DIVISOR)
            sample.update(label_dict)
        return sample

    def __len__(self):
        return len(self.image_files)

