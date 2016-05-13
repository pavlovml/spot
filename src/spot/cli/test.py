#! /usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

from spot.fast_rcnn.test import test_net
from spot.fast_rcnn.config import cfg
from spot.dataset import FasterRCNNDataset
from argparse import ArgumentParser

import caffe
import numpy as np
import pprint
import sys

def parse_args():
    """Parse input arguments"""
    parser = ArgumentParser(description='Train a Faster R-CNN network')

    parser.add_argument('-g', '--gpu', dest='gpu',
                        help='GPU device id to use',
                        default=0, type=int)

    parser.add_argument('-i', '--iterations', dest='iterations',
                        help='number of iterations to train',
                        default=40000, type=int)

    parser.add_argument('-s', '--seed', dest='seed',
                        help='fixed RNG seed',
                        default=None, type=int)

    parser.add_argument('--max-detections', dest='max_detections',
                        help='max number of detections per image',
                        default=10000, type=int)

    parser.add_argument('-v', '--visualize', dest='visualize',
                        help='visualize detections',
                        action='store_true')

    parser.add_argument('model', metavar='model',
                        help='model directory with test settings',
                        type=str)

    parser.add_argument('weights', metavar='weights',
                        help='weights for the model',
                        type=str)

    parser.add_argument('dataset', metavar='DATASET',
                        help='path to testing dataset',
                        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    cfg.GPU_ID = args.gpu

    print('Using config:')
    pprint.pprint(cfg)

    return args

def setup_caffe(gpu=0, seed=None):
    """Initializes Caffe's python bindings."""
    if seed:
        np.random.seed(seed)
        caffe.set_random_seed(seed)

    caffe.set_mode_gpu()
    caffe.set_device(gpu)

def run():
    args = parse_args()

    setup_caffe(gpu=args.gpu, seed=args.seed)

    dataset = FasterRCNNDataset(args.dataset)
    print 'Loaded dataset `{:s}` for testing ({:d} examples)'.format(dataset.name, len(dataset))

    test_file = '{:s}/test.prototxt'.format(args.model)
    net = caffe.Net(test_file, args.weights, caffe.TEST)
    all_boxes = test_net(net, dataset, max_per_image=args.max_detections, vis=args.visualize)
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """

    print 'done testing'
    sys.exit(0)
