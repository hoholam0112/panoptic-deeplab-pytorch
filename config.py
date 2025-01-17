""" Define default configuration using yacs """
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.GPUS = (0,)
_C.WORKERS = 0
# Logging frequency
_C.PRINT_FREQ = 20
# Checkpoint frequency
_C.CKPT_FREQ = 2000

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'panoptic_deeplab'
_C.MODEL.BN_MOMENTUM = 0.1

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# META could be
# resnet
# mobilenet_v2
# mnasnet
_C.MODEL.BACKBONE.META = 'resnet'

# NAME could be
# For resnet:
# 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
# For mobilenet_v2:
# 'mobilenet_v2'
# For mnasnet:
# 'mnasnet0_5', 'mnasnet0_75' (no official weight), 'mnasnet1_0', 'mnasnet1_3' (no official weight)
_C.MODEL.BACKBONE.NAME = "resnet50"
# Controls output stride
_C.MODEL.BACKBONE.DILATION = (False, False, False)
# pretrained backbone provided by official PyTorch modelzoo
_C.MODEL.BACKBONE.PRETRAINED = False

# Low-level feature key
# For resnet backbone:
# res2: 256
# res3: 512
# res4: 1024
# res5: 2048

# For mobilenet_v2 backbone:
# layer_4: 24
# layer_7: 32
# layer_14: 96
# layer_18: 320

# For mnasnet backbone:
# layer_9: 24 (0_5: 16)
# layer_10: 40 (0_5: 24)
# layer_12: 96 (0_5: 48)
# layer_14: 320 (0_5: 160)

# ---------------------------------------------------------------------------- #
# Decoder options
# ---------------------------------------------------------------------------- #
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.IN_CHANNELS = 2048
_C.MODEL.DECODER.FEATURE_KEY = 'res5'
_C.MODEL.DECODER.DECODER_CHANNELS = 256
_C.MODEL.DECODER.ATROUS_RATES = (3, 6, 9)

# TODO: pass these into the decoder.
_C.MODEL.DECODER.CONV_TYPE = 'depthwise_separable_conv'
_C.MODEL.DECODER.CONV_KERNEL = 5
_C.MODEL.DECODER.CONV_PADDING = 2
_C.MODEL.DECODER.CONV_STACK = 1

# ---------------------------------------------------------------------------- #
# Panoptic-DeepLab options
# ---------------------------------------------------------------------------- #
_C.MODEL.PANOPTIC_DEEPLAB = CN()
_C.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_CHANNELS = (1024, 512, 256)
_C.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_KEY = ('res4', 'res3', 'res2')
_C.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_CHANNELS_PROJECT = (128, 64, 32)
_C.MODEL.PANOPTIC_DEEPLAB.INSTANCE = CN()
_C.MODEL.PANOPTIC_DEEPLAB.INSTANCE.ENABLE = True
_C.MODEL.PANOPTIC_DEEPLAB.INSTANCE.LOW_LEVEL_CHANNELS_PROJECT = (64, 32, 16)
_C.MODEL.PANOPTIC_DEEPLAB.INSTANCE.DECODER_CHANNELS = 128
_C.MODEL.PANOPTIC_DEEPLAB.INSTANCE.HEAD_CHANNELS = 32
_C.MODEL.PANOPTIC_DEEPLAB.INSTANCE.ASPP_CHANNELS = 256
_C.MODEL.PANOPTIC_DEEPLAB.INSTANCE.NUM_CLASSES = (1, 2)
_C.MODEL.PANOPTIC_DEEPLAB.INSTANCE.CLASS_KEY = ('center', 'offset')

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NUM_CLASSES = 22
_C.DATASET.CROP_SIZE = (1081, 1921)
_C.DATASET.MIRROR = True
_C.DATASET.MIN_SCALE = 0.5
_C.DATASET.MAX_SCALE = 1.5
_C.DATASET.SCALE_STEP_SIZE = 0.1
_C.DATASET.MEAN = (0.485, 0.456, 0.406)
_C.DATASET.STD = (0.229, 0.224, 0.225)
_C.DATASET.SEMANTIC_ONLY = False
_C.DATASET.IGNORE_STUFF_IN_OFFSET = True
_C.DATASET.SMALL_INSTANCE_AREA = 4096
_C.DATASET.SMALL_INSTANCE_WEIGHT = 3

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.00005
_C.SOLVER.WEIGHT_DECAY = 0.0
# Weight decay of norm layers.
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
# Bias.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.OPTIMIZER = 'adam'
_C.SOLVER.ADAM_BETAS = (0.9, 0.999)
_C.SOLVER.ADAM_EPS = 1e-08

_C.SOLVER.LR_SCHEDULER_NAME = 'WarmupPolyLR'
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 0
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.POLY_LR_POWER = 0.9
_C.SOLVER.POLY_LR_CONSTANT_ENDING = 0

_C.SOLVER.CLIP_GRADIENTS = CN()
_C.SOLVER.CLIP_GRADIENTS.ENABLED = False
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.LOSS = CN()

_C.LOSS.SEMANTIC = CN()
_C.LOSS.SEMANTIC.NAME = 'hard_pixel_mining'
# TODO: make `ignore` more consistent
_C.LOSS.SEMANTIC.IGNORE = 255
_C.LOSS.SEMANTIC.REDUCTION = 'mean'
_C.LOSS.SEMANTIC.THRESHOLD = 0.7
_C.LOSS.SEMANTIC.MIN_KEPT = 100000
_C.LOSS.SEMANTIC.TOP_K_PERCENT = 0.2
_C.LOSS.SEMANTIC.WEIGHT = 1.0

_C.LOSS.CENTER = CN()
_C.LOSS.CENTER.NAME = 'mse'
_C.LOSS.CENTER.REDUCTION = 'none'
_C.LOSS.CENTER.WEIGHT = 200.0

_C.LOSS.OFFSET = CN()
_C.LOSS.OFFSET.NAME = 'l1'
_C.LOSS.OFFSET.REDUCTION = 'none'
_C.LOSS.OFFSET.WEIGHT = 0.01

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.IMS_PER_BATCH = 4
_C.TRAIN.MAX_ITER = 90000
_C.TRAIN.RESUME = False

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()

_C.TEST.GPUS = (0, )
_C.TEST.IMAGE_SIZE = (1080, 1920)

# -----------------------------------------------------------------------------
# POST PROCESSING
# Panoptic post-processing params
# -----------------------------------------------------------------------------
_C.POST_PROCESSING = CN()
_C.POST_PROCESSING.CENTER_THRESHOLD = 0.1
_C.POST_PROCESSING.NMS_KERNEL = 7
_C.POST_PROCESSING.TOP_K_INSTANCE = 200
_C.POST_PROCESSING.STUFF_AREA = 2048

def get_default_config():
    return _C.clone()
