# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from spot.config import cfg
from .gpu_nms import gpu_nms
from .cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    elif force_cpu:
        return cpu_nms(dets, thresh)
    else:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
