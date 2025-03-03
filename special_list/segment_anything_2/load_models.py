import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import json
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import hydra
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_module('sam2_configs', version_base='1.2')


def load_sam2(
    sam2_checkpoint,
    model_cfg = "sam2_hiera_l.yaml",
):
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
    return predictor

# load_sam2("/mnt/sdb/zpc/Dataset/models/SAM2/sam2_hiera_large.pt")