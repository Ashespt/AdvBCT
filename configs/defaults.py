import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']


# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# sampler
_C.DATA.CLASS_SAMPLER = False

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
# Mode specific
_C.MODEL.SPEC = CN(new_allowed=True)
# embedding size
_C.MODEL.EMB_DIM = 256
# architecture settings
_C.MODEL.ARCH = 'swsl_resnet18'
_C.MODEL.PRETRAINED = False
_C.MODEL.PRETRAINED_PATH = '' #None
_C.MODEL.NUM_CLASSES = 1000
# whether to use classification loss
_C.MODEL.USE_CLS = True

_C.OLD_MODEL = CN()
_C.OLD_MODEL.ARCH = 'swsl_resnet18'
_C.OLD_MODEL.PRETRAINED = False
_C.OLD_MODEL.PRETRAINED_PATH = '' #None
_C.OLD_MODEL.USE_CLS = True
_C.OLD_MODEL.NUM_CLASSES = 1000
_C.OLD_MODEL.MODEL_PATH = ''

_C.NEW_MODEL = CN()
_C.NEW_MODEL.ARCH = 'swsl_resnet18'
_C.NEW_MODEL.PRETRAINED = False
_C.NEW_MODEL.PRETRAINED_PATH = '' #None
_C.NEW_MODEL.NUM_CLASSES = 1000
_C.NEW_MODEL.USE_CLS = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.TYPE = 'base'
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.THRESHOLD = 0.5
_C.TRAIN.WARMUP_EPOCHS = 5
# Gradient accumulation steps
# Whether to use gradient checkpointing to save memory
# Frequency to save checkpoint
_C.TRAIN.SAVE_FREQ = 1
_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.PRINT_FREQ = 10
_C.TRAIN.INPUT_SIZE = 224
# Init param
_C.TRAIN.DATASET_TYPE = 'landmark'
_C.TRAIN.DATASET = 'ms1m'
_C.TRAIN.ROOT= './data/ms1m'
_C.TRAIN.FILE_DIR= './data/ms1m/ms1m_train_old_30percent_class.txt'
# the params of reid
_C.TRAIN.SAMPLER='softmax_triplet'
_C.TRAIN.NO_MARGIN=True
_C.TRAIN.METRIC_LOSS_TYPE='triplet'
_C.TRAIN.IF_LABELSMOOTH='off'
_C.TRAIN.ID_LOSS_WEIGHT=1.0
_C.TRAIN.TRIPLET_LOSS_WEIGHT=1.0
_C.TRAIN.TRIPLET_MARGIN=0.3
_C.TRAIN.CENTER_LOSS_WEIGHT=0.0005

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.BASE_LR = 8e-3
_C.TRAIN.LR_SCHEDULER.BIAS_LR_FACTOR = 2
_C.TRAIN.LR_SCHEDULER.WEIGHT_DECAY_BIAS = 1e-4
_C.TRAIN.LR_SCHEDULER.WEIGHT_DECAY = 1e-4
_C.TRAIN.LR_SCHEDULER.LARGE_FC_LR = False
_C.TRAIN.LR_SCHEDULER.MOMENTUM = 0.9
_C.TRAIN.LR_SCHEDULER.CENTER_LR = 0.5
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.BATCH_SIZE = 64






# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = '' #None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

_C.LOSS = CN()
_C.LOSS.TYPE = 'softmax'
_C.LOSS.SCALE = 30.0
_C.LOSS.MARGIN = 0.3


_C.COMP_LOSS = CN()
_C.COMP_LOSS.TYPE = 'bct'
_C.COMP_LOSS.ELASTIC_BOUNDARY = False
_C.COMP_LOSS.TEMPERATURE = 0.01
_C.COMP_LOSS.TRIPLET_MARGIN = 0.8
_C.COMP_LOSS.TOPK_NEG = 10
_C.COMP_LOSS.WEIGHT = 1.0
_C.COMP_LOSS.DISTILLATION_TEMP = 0.01
_C.COMP_LOSS.FOCAL_BETA = 5.0
_C.COMP_LOSS.FOCAL_ALPHA = 1.0

_C.UPGRADE_LOSS = CN()
_C.UPGRADE_LOSS.TYPE = 'base'
_C.UPGRADE_LOSS.WEIGHT = 1

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Eval settings
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# Whether to use center crop when testing
_C.EVAL.SAVE_DIR = './logs/eval/tmp_eval'
_C.EVAL.DEVICE = 0
_C.EVAL.LOG_DIR = './logs/eval'
_C.EVAL.DATASET_TYPE = 'landmark'
_C.EVAL.DATASET_NAME = 'gldv2'
_C.EVAL.BCT_TYPE = 'base model'
_C.EVAL.ROOT= 'base model'
_C.EVAL.DATASET= 'roxford5k'
_C.EVAL.PRINT_FREQ = 1
_C.EVAL.OLD_MODEL_PATH = ''
_C.EVAL.NEW_MODEL_PATH = ''
#old model center
_C.EVAL.OLD_SAVE_FILE = ''
_C.EVAL.HOT_REFRESH = False


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
_C.USE_AMP = False
_C.DIST_BACKEND = 'nccl'
_C.DIST_URL = 'tcp://127.0.0.1:23456'
_C.MULTIPROCESSING_DISTRIBUTED = False
_C.DISTRIBUTED = False
_C.WORLD_SIZE = 1
# Path to output folder, overwritten by command line argument
_C.OUTPUT = 'output'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to logging info
_C.PRINT_FREQ = 100
# Fixed random seed
_C.SEED = 1234
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Debug only so that skip dataloader initialization, overwritten by command line argument
_C.DEBUG_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
_C.RANK = 0
_C.DEVICE = 'cuda'
_C.GATHER_ALL = True



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)
    config.merge_from_list(args.opts)
    config.defrost()

    # merge from specific arguments
    # if args.output:
    #     config.OUTPUT = args.output

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.ARCH, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # print('config', config)
    update_config(config, args)
    return config
