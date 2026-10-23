#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode


# Global config object
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.DBG = False
_C.OUTPUT_DIR = "./output"
_C.RUN_N_TIMES = 1
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.SEED = 0

# Model options
_C.MODEL = CfgNode()
_C.MODEL.TRANSFER_TYPE = "end2end"  # linear or end2end
_C.MODEL.WEIGHT_PATH = ""  # if resume from some checkpoint file
_C.MODEL.ALLOW_LEGACY_CHECKPOINT = False
_C.MODEL.SAVE_CKPT = False

_C.MODEL.MODEL_ROOT = "model/pretrained"

_C.MODEL.TYPE = "vit"
_C.MODEL.MLP_NUM = 0

# CADLGI configuration shared by Rep and Swin.
_C.MODEL.CADLGI = CfgNode()
_C.MODEL.CADLGI.BACKBONE_TYPE = "repvit_m1_0"
_C.MODEL.CADLGI.NUM_HEADS = 4
_C.MODEL.CADLGI.USE_ALGF = True
_C.MODEL.CADLGI.ALGF_GATE_MODE = "joint"  # joint, sample, spatial, static, concat
_C.MODEL.CADLGI.ALGF_VERSION = 2  # 1: static fusion control; 2: ALGF
_C.MODEL.CADLGI.USE_SPATIAL_ADAPTIVE = True
_C.MODEL.CADLGI.USE_CADPA = True
_C.MODEL.CADLGI.MAX_PROTOTYPES = 32
_C.MODEL.CADLGI.MIN_ACTIVE_RATIO = 0.0625
_C.MODEL.CADLGI.DROP_PATH_RATE = 0.0  # RepViT implementation has no stochastic-depth path
_C.MODEL.CADLGI.LABEL_SMOOTH = 0.0
_C.MODEL.CADLGI.HEAD_DROP = 0.2
_C.MODEL.CADLGI.USE_PRETRAINED = True
_C.MODEL.CADLGI.PRETRAINED_PATH = "auto"

_C.SOLVER = CfgNode()
_C.SOLVER.LOSS = "softmax"
_C.SOLVER.LABEL_SMOOTH = 0.0  # shared loss for every backbone and ablation
_C.SOLVER.LOSS_ALPHA = 0.01

_C.SOLVER.OPTIMIZER = "sgd"  # or "adamw"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.BIAS_MULTIPLIER = 1.0

_C.SOLVER.PATIENCE = 300


_C.SOLVER.SCHEDULER = "cosine"

_C.SOLVER.BASE_LR = 0.01

_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.TOTAL_EPOCH = 30
_C.SOLVER.LOG_EVERY_N = 1000


_C.SOLVER.DBG_TRAINABLE = False # if True, will print the name of trainable params
_C.SOLVER.SELECTION_METRIC = "top1"  # fixed before training; validation only
_C.SOLVER.AMP = True
_C.SOLVER.TF32 = True
_C.SOLVER.CUDNN_BENCHMARK = True

# Dataset options
_C.DATA = CfgNode()

_C.DATA.NAME = "barley"
_C.DATA.DATAPATH = "../split_cropped_dataset"
_C.DATA.INDEX_DIR = ""  # optional derived protocol indices, images remain at DATAPATH
_C.DATA.FEATURE = "sup_vitb16_imagenet21k"  # e.g. inat2021_supervised

_C.DATA.PERCENTAGE = 1.0
_C.DATA.NUMBER_CLASSES = 6
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"

_C.DATA.CROPSIZE = 224  # or 384

_C.DATA.NO_TEST = True  # leave the test split sealed during model development
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True
# Enable the configured training augmentation.
_C.DATA.USE_STRONG_AUGMENTATION = False

_C.DIST_BACKEND = "nccl"
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
