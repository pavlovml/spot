#! /usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from spot.dataset import FasterRCNNDataset
from spot.fast_rcnn.config import cfg
from spot.fast_rcnn.test import test_net
import caffe, sys
import numpy as np

def add_subparser(parent):
    parser = parent.add_parser('test', help='Train a Faster R-CNN network')
    parser.set_defaults(func=run)

    parser.add_argument(
            'model', metavar='MODEL',
            help='model directory with test settings',
            type=file)

    parser.add_argument(
            'weights', metavar='WEIGHTS',
            help='weights for the model',
            type=file)

    parser.add_argument(
            'dataset', metavar='DATASET',
            help='path to testing dataset',
            type=str)

    testing_group = parser.add_argument_group('testing arguments')

    testing_group.add_argument(
            '-g', '--gpu', dest='gpu',
            help='GPU device id to use',
            default=0, type=int)

    testing_group.add_argument(
            '-i', '--iterations', dest='iterations',
            help='number of iterations to train',
            default=40000, type=int)

    testing_group.add_argument(
            '-s', '--seed', dest='seed',
            help='fixed RNG seed',
            default=None, type=int)

    testing_group.add_argument(
            '--max-detections', dest='max_detections',
            help='max number of detections per image',
            default=10000, type=int)

def setup_caffe(gpu=0, seed=None):
    """Initializes Caffe's python bindings."""
    cfg.GPU_ID = gpu

    if seed:
        np.random.seed(seed)
        caffe.set_random_seed(seed)

    caffe.set_mode_gpu()
    caffe.set_device(gpu)

def run(args):
    setup_caffe(gpu=args.gpu, seed=args.seed)

    dataset = FasterRCNNDataset(args.dataset)
    print 'Loaded dataset `{:s}` for testing ({:d} examples)'.format(dataset.name, len(dataset))

    net = caffe.Net(args.model.name, args.weights.name, caffe.TEST)

    # all_boxes is a list of length number-of-classes.
    # Each list element is a list of length number-of-images.
    # Each of those list elements is either an empty list []
    # or a numpy array of detection.
    #
    # all_boxes[class][image] = [] or np.array of shape #dets x 5
    all_boxes = test_net(net, dataset, max_per_image=args.max_detections, vis=False)

    print 'done testing'
    sys.exit(0)
