""" Generate rgb mask in which label and instance information is encoded as RGB value. """
import os, argparse
from glob import glob
from PIL import Image
import numpy as np
import cv2

# Custom
from data import get_label_divisor, class_name_to_id

_MASK_DIR = './dataset/train_mask'
_ROOT_DIR = './dataset/train'

def create_mask(img, img_elem, label_divisor):
    mask = np.zeros_like(img, dtype=np.uint8)
    instance_id = 1
    for polygon_elem in img_elem.findall('./polygon'):
        # Read and process class_name
        class_name = polygon_elem.attrib['label']
        if class_name != 'bike_lane': # class 'bike_lane' has no attribute
            if len(polygon_elem.findall('attribute')) == 0: # Exception handling
                continue
            class_name += '_'+polygon_elem.findall('attribute')[0].text
        cat_id = class_name_to_id(class_name)
        # Read and decode polygon string to np.array
        polygon_str = polygon_elem.attrib['points']
        points = decode_polygon(polygon_str)
        # Encode instance id
        panoptic_id = cat_id*label_divisor + instance_id
        rgb = id2rgb(panoptic_id) # encode instance id to RGB value 
        cv2.fillPoly(mask, [points], rgb) # assing RGB value to label 
        instance_id += 1
    return mask

def create_segments(mask, label_divisor, rgb2id):
    segments = []
    mask_id = rgb2id(mask)
    panoptic_ids = np.unique(mask_id)
    for panoptic_id in panoptic_ids:
        seg = {}
        seg['category_id'] = panoptic_id // label_divisor
        # Encode instance id
        seg['id'] = panoptic_id % label_divisor
        # Append seg to segments
        seg['iscrowd'] = False
        segments.append(seg)
    return segments

def main():
    """ Generate mask to make label reading process faster, which results in fast training.
        panoptic_id = class_id * label_divisor + instance_id as 256-base number.
    """
    label_divisor = get_label_divisor()
    image_files = [] # list of file names
    image_elements = []
    for d in sorted(os.listdir(_ROOT_DIR)):
        files = sorted(glob(os.path.join(root, d, '*.jpg')))
        image_files += files
        # read xml files and collect image elements
        xml_path = glob(os.path.join(root, d, '*.xml'))[0]
        root_element = elemTree.parse(xml_path)
        image_element = root_element.findall('./image')
        image_elements += image_element

    i = 1
    for filename, elem in zip(image_files, image_elements):
        mask_filename = '/'.join(filename.split('/')[3:]).split('.')[0] + '.png'
        mask_path = os.path.join(_MASK_DIR, mask_filename)
        # create Mask sub directory
        os.makedirs('/'.join(mask_path.split('/')[:-1]), exist_ok=True)
        # create mask
        img = np.array(Image.open(filename))
        mask = create_mask(img, elem, label_divisor)
        mask.save(mask_path)
        print('{}-th Mask is created and saved: {}'.format(i, mask_path))
        i += 1


if __name__ == '__main__':
    main()

