""" function to process model output for evaluation format """
import numpy as np
import torch
import torch.nn.functional as F

from .semantic_post_processing import get_semantic_segmentation
from .instance_post_processing import get_instance_segmentation

def get_ins_list(semantic_pred,
                 center_pred,
                 offset_pred,
                 thing_list,
                 image_size=(1080, 1920),
                 threshold=0.1,
                 nms_kernel=7,
                 top_k=None):
    """ get instance list from prediction results.

    Args:
        sementic_pred: tensor of size [1, C, H, W]. sementic segmentation prediction.
        center_pred: tensor of size [1, 1, H, W]. center heatmap prediction.
        offset_pred: tensor of size [1, 2, H, W]. offset prediction.
        thing_list: List of thing class ids (int). things can be instance and belongs to foregound.
        image_size: (height, width). standard size of image for evaluation.
        threshold: float. threshold for center_pred activation.
        nms_kernel: int. max pooling kernel size for filtering center activations.
        top_k: int. number of center points to be preserved from predicted center heatmap.

    Returns:
        eval_format: list of dictionary. e.g.
            {'class_id': 13, 'mask': nparray of size (h, w) , 'score': 0.9876}


    Raises:
        AssertionError: check prediction maps' dimension.
    """
    # Check argument validity
    assert semantic_pred.dim() == 4 and semantic_pred.size(0) == 1
    assert center_pred.dim() == 4 and center_pred.size(0) == 1
    assert offset_pred.dim() == 4 and offset_pred.size(0) == 1

    sem_soft = F.softmax(semantic_pred, dim=1) # Normalize prediction scores
    # Resize prediction maps for evaluation 
    h, w = image_size
    if h != sem_soft.size(2) or w != sem_soft.size(3):
        sem_soft = F.interpolate(
                sem_soft, size=(h, w),
                mode='bilinear', align_corners=False
            )
    if h != center_pred.size(2) or w != center_pred.size(3):
        center_pred = F.interpolate(
                center_pred, size=(h, w),
                mode='bilinear', align_corners=False
            )
    if h != offset_pred.size(2) or w != offset_pred.size(3):
        offset_pred = F.interpolate(
                offset_pred, size=(h, w),
                mode='bilinear', align_corners=False
            )

    # sem_soft [1, C, H, W] -> sem_hard [1, H, W]
    sem_hard = get_semantic_segmentation(sem_soft)
    ins_seg, center_points = get_instance_segmentation(
            sem_hard, center_pred, offset_pred, thing_list,
            threshold, nms_kernel, top_k)

    # select instance's class label by majority bote.
    instance_ids = torch.unique(ins_seg)
    ins_list = []
    for ins_id in instance_ids:
        instance = {}
        if ins_id == 0:
            continue
        #majority voting
        ins_mask = (ins_seg == ins_id)
        class_id, _ = torch.mode(sem_hard[ins_mask].view(-1, ))
        instance['class_id'] = class_id.item()
        # get polygon from binary instance mask
        instance['mask'] = ins_mask.squeeze(0).detach().cpu().numpy()
        # Compute confidence score
        score = torch.mean(sem_soft[:, class_id, :, :][ins_mask])
        instance['score'] = score.item()
        ins_list.append(instance)

    if not ins_list:
        raise RuntimeError('mage has no detected instance.')
    return ins_list

