""" function to process model output for evaluation format """
import numpy as np
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
    assert semantic_pred.dim() == 4
    assert center_pred.dim() == 4
    assert offset_pred.dim() == 4

    sem_soft = F.softmax(semantic_pred, dim=1) # Normalize prediction scores
    # Resize prediction maps for evaluation 
    h, w = image_size
    sem_soft = F.interpolate(sem_soft, size=(h, w), mode='bilinear')
    ctr_hmp = F.interpolate(center_pred, size=(h, w), mode='bilinear')
    off = F.interpolate(offset_pred, size=(h, w), mode='bilinear')

    # sem_soft [1, C, H, W] -> sem_hard [1, H, W]
    sem_hard = get_semantic_segmentation(sem_soft)
    ins_seg, center_points = get_instance_segmentation(
            sem_hard, ctr_hmp, off, thing_list,
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
        instance['mask'] = ins_mask.squeeze(0).cpu().numpy()
        # Compute confidence score
        instance['score'] = torch.mean(sem_soft.squeeze(0)[class_id]).item()
        ins_list.apppend(instance)

    if not ins_list:
        raise RuntimeError('mage has no detected instance.')
    return ins_list

