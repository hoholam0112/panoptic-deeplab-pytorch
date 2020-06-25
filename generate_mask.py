""" Generate rgb mask in which label and instance information is encoded as RGB value. """
import os, argparse
from glob import glob


_MASK_DIR = './dataset/train_mask'
_ROOT_DIR = './dataset/train'

def main():
    """ Generate mask to make label reading process faster, which results in fast training.
        panoptic_id = class_id * label_divisor + instance_id as 256-base number.
    """

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

    for filename, elem in zip(image_files, image_elements):
        mask_filename = '/'.join(filename.split('/')[3:]).split('.')[0] + '.png'
        mask_path = os.path.join(_MASK_DIR, mask_filename)
        print(mask_path)
        os.makedirs('/'.join(mask_path.split('/')[:-1]), exist_ok=True)


        mask = np.zeros_like([1080, 1920, 3], dtype=np.uint8)


if __name__ == '__main__':
    main()

