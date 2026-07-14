import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from ritm_interactive_segmentation.isegm.data.datasets import *
from ritm_interactive_segmentation.isegm.model.losses import *
from ritm_interactive_segmentation.isegm.data.transforms import *
from ritm_interactive_segmentation.isegm.engine.trainer import ISTrainer
from ritm_interactive_segmentation.isegm.model.metrics import AdaptiveIoU
from ritm_interactive_segmentation.isegm.data.points_sampler import MultiPointSampler
from ritm_interactive_segmentation.isegm.utils.log import logger
from ritm_interactive_segmentation.isegm.model import initializer

from ritm_interactive_segmentation.isegm.model.is_hrnet_model import HRNetModel
from ritm_interactive_segmentation.isegm.model.is_deeplab_model import DeeplabModel