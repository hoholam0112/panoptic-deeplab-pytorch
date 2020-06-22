# ------------------------------------------------------------------------------
# Builds transformation for both image and labels.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from . import transforms as T


def build_transforms(dataset, is_train=True):
    min_scale = dataset.min_scale
    max_scale = dataset.max_scale
    scale_step_size = dataset.scale_step_size
    crop_h = dataset.crop_h
    crop_w = dataset.crop_w
    pad_value = dataset.pad_value
    ignore_label = dataset.label_pad_value
    flip_prob = 0.5 if dataset.mirror else 0
    mean = dataset.mean
    std = dataset.std

    if is_train:
        transforms = T.Compose(
            [
                T.RandomScale(min_scale,
                              max_scale,
                              scale_step_size),
                T.RandomCrop(crop_h,
                             crop_w,
                             pad_value,
                             ignore_label,
                             random_pad=is_train),
                T.RandomHorizontalFlip(flip_prob),
                T.ToTensor(),
                T.Normalize(mean,
                            std)
            ]
        )
    else:
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean,
                            std)
            ]
        )

    return transforms
