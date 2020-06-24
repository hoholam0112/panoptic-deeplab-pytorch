""" panoptic-deeplab for road condition segmentation """
import os, argparse
import torch
import numpy as np
import random
import glob

# Custom libs
from config import get_default_config
from data import BaseDataset
from model import build_segmentation_model_from_cfg
from utils import AverageMeter

def save(save_path, model, optimizer, lr_scheduler, step):
    """ save training state as checkpoint file """
    checkpoint = {'model_state_dict' : model.state_dict(),
                  'optimizer_state_dict' : optimizer.state_dict(),
                  'lr_scheduler_state_dict' : lr_scheduler.state_dict(),
                  'step' : step}
    # change postfix before saving
    new_save_path = save_path.split('-')[0] + '-{:d}.pt'.format(step)
    torch.save(checkpoint, save_path)

def load(checkpoint, model, optimizer=None, lr_scheduler=None):
    """ Load checkpoint """
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Panoptic Deeplab for road condition recognition.')
    parser.add_argument('--root_dir', help='root directory of dataset.', required=True)
    parser.add_argument('--config_file', help='.yaml file path for configuration.')
    parser.add_argument('--resume', help='whether to resume train.', action='store_true')
    parser.add_argument('--test', help='whether to train or test.', action='store_true')
    parser.add_argument('--output_path', help='When test is True, path of prediction xml file.')
    args = parser.parse_args()

    root_dir = args.root_dir
    config_file = args.config_file
    resume = args.resume
    test = args.test
    output_path = args.output_path

    # Load and update configuration 
    cfg = get_default_config()
    if config_file:
        cfg.merge_from_file(config_file)
    cfg.freeze()

    # Set device to be used. 
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        raise RuntimeError('cuda is not available.')

    # Fixed random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # set training
    training = True
    if test:
        training = False
        if output_path is None:
            raise ValueError('output_path should be passed when test is True.')
    # set batch_size
    batch_size = 1
    if training:
        batch_size = cfg.TRAIN.IMS_PER_BATCH

    # load checkpoint
    save_path = './checkpoint_dir'
    os.makedirs(save_path, exist_ok=True)
    filename = config_file.split('/')[-1].split('.')[0]
    save_path = os.path.join(checkpoint_dir, filename)
    save_list = glob.glob(save_path + '*')
    if len(save_list) == 0:
        save_path += '-0.pt'
    else:
        save_path = sorted(save_list)[-1]

    checkpoint = None
    if training:
        if os.path.exists(save_path):
            if resume:
                checkpoint = torch.load(save_path)
            else:
                raise ValueError(
                        """checkpoint file already exists.
                           training seems to be stopped.
                           add --resume argument and try again."""
                        )
    else:
        checkpoint = torch.load(save_path)

    # Load dataset
    dataset = BaseDataset(root=root_dir,
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
            dataset, batch_size=batch_size,
            shuffle=training, pin_memory=True, num_workers=cfg.WORKERS
        )

    # Build model
    model = build_segmentation_model_from_cfg(cfg)
    model.to(device)

    time_meter = AverageMeter()
    if training: # Train
        step = 0
        max_iter = cfg.TRAIN.MAX_ITER
        optimizer = torch.optim.Adam(
                model.parameters(), cfg.SOLVER.BASE_LR,
                betas=cfg.SOLVER.ADAM_BETAS, eps=cfg.SOLVER.ADAM_EPS
            )
        optimizer.to(device)
        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA
            )
        if checkpoint: # restore training state
            step = checkpoint['step']
            load(checkpoint, model, optimizer, lr_scheduler)

        # Trainig loop
        model.train()
        iterator = iter(loader)
        while step < max_iter:
            try:
                start_time = time.time() # check start time
                batch = next(iterator)
                # Copy tensors to device
                image = batch['image'].to(device)
                target_keys = ['semantic', 'semantic_weights',
                               'center', 'center_weights',
                               'offset', 'offset_weights']
                targets = {k : batch[k].to(device) for k in target_keys}

                # forward pass
                output = model(image, targets)
                # backward pass
                loss = output['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                elapsed_time = time.time() - start_time # check elaped time
                time_meter.update(elapsed_time)
                step += 1 # increment step
                # Print summary logs
                if step % cfg.PRINT_FREQ == 0:
                    print('{:d}/{:d} -- {:d}s'.format(step, max_iter, int(time_meter.sum)))
                    for k, v in model.loss_meter_dict.items():
                        print('{}: {.6f}  |  '.format(k, v.avg), end='')
                    print('')
                    time_meter.reset()
                # Save model
                if step % cfg.CKPT_FREQ == 0:
                    save(save_path, model, optimizer, lr_scheduler, step)
                    print('model saved.')

            except StopIteration:
                iterator = iter(loader)
    else: # Test
        raise NotImplementedError('test part is not implemented yet.')



