# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Faster R-CNN."""

import os.path as osp
import sys

def add_path(*parts):
    this_dir = osp.dirname(__file__)
    path = osp.join(this_dir, '..', *parts)
    if path not in sys.path:
        sys.path.insert(0, path)

add_path('gibraltar', 'current', 'python')
add_path('src')
