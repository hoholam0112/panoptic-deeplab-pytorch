""" panoptic-deeplab for road condition segmentation """
import argparse
import torch


# Custom libs
from config import get_default_config
from data import BaseDataset
from model import build_segmentation_model_from_cfg


def save_model(model, save_path):
    pass

def load_model(save_path):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Panoptic Deeplab for road condition recognition.')
    parser.add_argument('--root_dir', help='root directory of dataset.', required=True)
    parser.add_argument('--config_file', help='.yaml file path for configuration.')
    parser.add_argument('--resume', help='whether to resume train.', action='store_true')
    parser.add_argument('--test', help='whether to train or test.', action='store_true')
    args = parser.parse_args()

    # Load and update configuration 
    cfg = get_default_config()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # Set device to be used. 
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        raise RuntimeError('cuda is not available.')

    # Set training
    training = True
    if args.test:
        training = False

    # Load dataset
    dataset = BaseDataset(root=args.root_dir,
                          training=training,
                          crop_size=cfg.DATASET.CROP_SIZE,
                          mirror=cfg.DATASET.MIRROR,
                          min_scale=cfg.DATASET.MIN_SCALE,
                          max_scale=cfg.DATASET.MAX_SCALE,
                          scale_step_size=cfg.DATASET.SCALE_STEP_SIZE,
                          mean=cfg.DATASET.MEAN,
                          std=cfg.DATASET.STD,
                          semantic_only=cfg.DATASET.SEMANTIC_ONLY,
                          ignore_stuff_in_offset=cfg.DATASET.IGNORE_STUFF_IN_OFFSET,
                          small_instance_area=cfg.DATASET.SMALL_INSTANCE_AREA,
                          small_instance_weight=cfg.DATASET.SMALL_INSTANCE_WEIGHT)

    loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH,
            shuffle=training, pin_memory=True, num_workers=cfg.WORKERS
        )

    # Build model
    model = build_segmentation_model_from_cfg(cfg)
    model.to(device)

    # Trainig loop
    iterator = iter(loader)
    batch = next(iterator)
    image = batch['image'].to(device)
    target_keys = ['semantic', 'semantic_weights',
                   'center', 'center_weights',
                   'offset', 'offset_weights']
    targets = {k : batch[k].to(device) for k in target_keys}




