""" panoptic-deeplab for road condition segmentation """
import os, argparse, time
import torch
import numpy as np
import random
import glob
from torch.utils.tensorboard import SummaryWriter
from imantics import Polygons
import xml.etree.ElementTree as elemTree
import torch.nn.functional as F
import cv2

# Custom libs
from config import get_default_config
from data import BaseDataset, get_thing_list, class_id_to_name
from model import build_segmentation_model_from_cfg
from utils import AverageMeter, get_image_id, points_to_str, NoiseRemover
from post_processing import get_semantic_segmentation

def save(save_path, model, optimizer, lr_scheduler, step):
    """ save training state as checkpoint file """
    checkpoint = {'model_state_dict' : model.state_dict(),
                  'optimizer_state_dict' : optimizer.state_dict(),
                  'lr_scheduler_state_dict' : lr_scheduler.state_dict(),
                  'step' : step}
    # change postfix before saving
    new_save_path = save_path.split('-')[0] + '-{:05d}.pt'.format(step)
    torch.save(checkpoint, new_save_path)

def load(checkpoint, model, optimizer=None, lr_scheduler=None):
    """ Load checkpoint """
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

def get_lr(cfg, optimizer, step):
    """ get learning for this step """
    if step < 30000:
        optimizer.param_groups[0]['lr'] = cfg.SOLVER.BASE_LR
    elif step >= 30000 and step < 60000:
        optimizer.param_groups[0]['lr'] = cfg.SOLVER.BASE_LR*0.1
    else:
        optimizer.param_groups[0]['lr'] = cfg.SOLVER.BASE_LR*0.01

class InstanceDetector:
    """ Instance detector """
    def __init__(self,
                 device,
                 num_classes=22,
                 ignore_label=0,
                 min_instance_area=0.02,
                 remover_kernel_size=7,
                 erode_iter=1,
                 dilate_iter=3):
        """ initialize an instance detector """
        self.device = device
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.min_instance_area = min_instance_area
        self.noise_remover = NoiseRemover(
                remover_kernel_size, erode_iter, dilate_iter)

    def detect_polygons(self, class_mask, min_area):
        """ Detect polygons from binary mask whose area is greater than min_area.
        Args:
            class_mask: np.array of shape [H, W]. mask of specific class

        Yields:
            ins_mask: np.array of the same shape with mask. detected instance mask
            points: np.array of shape (N, 2) sequeze of points.
        """
        points_list = Polygons.from_mask(class_mask)
        for points in points_list:
            points = np.array([[points[2*i], points[2*i+1]] for i in range(len(points)//2)])
            if points.shape[0] < 3:
                continue
            ins_mask = np.zeros_like(class_mask, dtype=np.uint8)
            cv2.fillPoly(ins_mask, [points], 255)
            ins_mask_area = np.sum(ins_mask != 0)
            # Remove false positive detections in background
            if np.sum(class_mask*ins_mask.astype(np.bool)) < int(ins_mask_area * 0.5):
                continue
            if ins_mask_area > min_area:
                yield ins_mask, points

    def __call__(self, sem_pred, raw_size):
        """ detect instances. Returns list of dict(class_id=int, points=np.array, score=float)
        Args:
            sem_pred (tensor of shape [1, K, H, W]): semantic segmentation prediction
            raw_size (tuple of int): (height, width). raw image size points is re-scaled to raw_size.
        """
        assert sem_pred.dim() == 4 and sem_pred.size(0) == 1
        # assign semantic label
        sem_hard = get_semantic_segmentation(sem_pred)
        sem_pred = F.softmax(sem_pred, dim=1)
        sem_hard.squeeze_(0)
        sem_pred.squeeze_(0)
        # Define minimal area size for instance
        h_raw, w_raw = raw_size
        _, h, w  = sem_pred.shape
        total_area = h*w
        min_instance_area = int(self.min_instance_area * total_area)
        ins_list = []
        for cat_id in range(self.num_classes):
            if cat_id == self.ignore_label:
                continue
            class_mask = (sem_hard == cat_id)
            class_mask_numpy = class_mask.cpu().numpy()
            cat_area = torch.sum(class_mask).item()
            if cat_area < min_instance_area:
                continue
            # Remove noisy part and smoothing mask
            mask_denoised = self.noise_remover(class_mask_numpy)
            for ins_mask, points in self.detect_polygons(mask_denoised, min_instance_area):
                instance = {}
                instance['class_name'] = class_id_to_name(cat_id)
                # Compute confidence score
                ins_mask_tensor = torch.tensor(ins_mask, dtype=torch.bool).to(self.device)
                score_sum = torch.sum(sem_pred[cat_id, :, :]*ins_mask_tensor)
                score_mean = score_sum/torch.sum(ins_mask_tensor)
                instance['score'] = score_mean.item()
                # scale points to raw size
                w_factor = (w_raw-1)/float(w-1)
                h_factor = (h_raw-1)/float(h-1)
                points_scale = np.zeros_like(points)
                for i in range(points.shape[0]):
                    points_scale[i, 0] = int(w_factor*points[i, 0])
                    points_scale[i, 1] = int(h_factor*points[i, 1])
                instance['points'] = points_scale
                ins_list.append(instance)
        return ins_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Panoptic Deeplab for road condition recognition.')
    parser.add_argument('--root_dir', help='root directory of dataset.', required=True)
    parser.add_argument('--config_file', help='.yaml file path for configuration.')
    parser.add_argument('--resume', help='whether to resume train.', action='store_true')
    parser.add_argument('--test', help='whether to train or test.', action='store_true')
    parser.add_argument('--step_ckp', help='step number of checkpoint.', action='store_true')
    args = parser.parse_args()

    root_dir = args.root_dir
    config_file = args.config_file
    resume = args.resume
    test = args.test
    step_ckp = args.step_ckp

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
    # set batch_size
    batch_size = 1
    if training:
        batch_size = cfg.TRAIN.IMS_PER_BATCH

    # load checkpoint
    save_path = './checkpoint_dir'
    os.makedirs(save_path, exist_ok=True)
    config_filename = config_file.split('/')[-1].split('.')[0]
    save_path = os.path.join(save_path, config_filename)
    save_list = glob.glob(save_path + '*')
    if len(save_list) == 0:
        save_path += '-0.pt'
    else:
        save_path = sorted(save_list)[-1] # Load latest checkpoint
        if step_ckp:
            save_path += '-{:05d}.pt'.format(step_ckp)

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
        writer = None

        optimizer = torch.optim.Adam(
                model.parameters(), cfg.SOLVER.BASE_LR,
                betas=cfg.SOLVER.ADAM_BETAS, eps=cfg.SOLVER.ADAM_EPS
            )
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
                #lr_scheduler.step()
                get_lr(cfg, optimizer, step)

                elapsed_time = time.time() - start_time # check elaped time
                time_meter.update(elapsed_time)
                step += 1 # increment step
                # Print summary logs
                if step % cfg.PRINT_FREQ == 0:
                    print('{:d}/{:d} -- {:d}s'.format(step, max_iter, int(time_meter.sum)))
                    for k, v in model.loss_meter_dict.items():
                        print('{}: {:.6f}  |  '.format(k, v.avg), end='')
                    print('')
                    time_meter.reset()
                    # Tensorboard summary writer
                    if writer is None:
                        writer = SummaryWriter('./summary_dir/{}'.format(config_filename))
                    for k, v in model.loss_meter_dict.items():
                        writer.add_scalar('Loss/{}'.format(k), v.avg, step)
                    writer.flush()

                # Save model
                if step % cfg.CKPT_FREQ == 0:
                    save(save_path, model, optimizer, lr_scheduler, step)
                    print('model saved.')
            except StopIteration:
                iterator = iter(loader)
        writer.close()
    else: # Test
        load(checkpoint, model) # Load saved model
        # Create instance detector object
        detector = InstanceDetector(device,
                                    num_classes=cfg.DATASET.NUM_CLASSES,
                                    ignore_label=0,
                                    min_instance_area=0.02,
                                    remover_kernel_size=7,
                                    erode_iter=1,
                                    dilate_iter=3)
        pred_xml = elemTree.Element('predictions') # .xml file to save prediction
        pred_xml.text = '\n  '
        model.eval()
        start_time = time.time()
        for i, batch in enumerate(loader):
            with torch.no_grad():
                image = batch['image'].to(device)
                output = model(image)
                h, w = list(batch['raw_size'].squeeze(0).numpy())
                ins_list = detector(output['semantic'].to(device), raw_size=(h, w))

            idx = batch['dataset_index'].squeeze(0).item()
            image_id = get_image_id(dataset.image_files[idx])
            # Create sub-element image 
            xml_image = elemTree.SubElement(pred_xml, 'image')
            xml_image.attrib['name'] = image_id
            xml_image.text = '\n    '
            # Write detected instance in .xml file 
            for j, ins in enumerate(ins_list):
                # Creatae sub-element predict 
                xml_predict = elemTree.SubElement(xml_image, 'predict')
                xml_predict.tail = '\n    '
                xml_predict.attrib['class_name'] = ins['class_name']
                # binary mask to polygons
                points_str = points_to_str(ins['points'])
                xml_predict.attrib['polygon'] = points_str
                xml_predict.attrib['score'] = str(float(ins['score']))
            xml_predict.tail = '\n  '
            xml_image.tail = '\n  '
            elapsed_time = time.time() - start_time
            print('\r{:05d}/{:05d} -- {:d}s'.format(i+1, len(dataset), int(elapsed_time)), end='')
        print('')
        xml_image.tail = '\n'

        pred_xml = elemTree.ElementTree(pred_xml)
        split = root_dir.split('/')[-1]
        output_dir = './predictions/{}/{}'.format(split, config_filename)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'pred-{:05d}.xml'.format(checkpoint['step']))
        pred_xml.write(output_path)
        print('xml file is saved at: {}'.format(output_path))


