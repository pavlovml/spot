#! /usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

from spot.fast_rcnn.train import FasterRCNNSolver
from spot.fast_rcnn.config import cfg
from spot.utils.mkdirp import mkdirp
from spot.dataset import FasterRCNNDataset
from argparse import ArgumentParser

import caffe
import numpy as np
import pprint
import spot.roi_data_layer.roidb as rdl_roidb
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

    parser.add_argument('-f', '--flipped', dest='flipped',
                        help='include flipped images in training dataset',
                        action='store_true')

    parser.add_argument('-o', '--output', dest='output',
                        help='directory to save snapshots to',
                        default='output', type=str)

    parser.add_argument('--snapshot-iterations', dest='snapshot_iterations',
                        help='number of iterations between snapshots',
                        default=10000, type=int)

    parser.add_argument('model', metavar='model',
                        help='model directory with solver/test/train settings and weights',
                        type=str)

    parser.add_argument('dataset', metavar='DATASET',
                        help='path to training dataset',
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
    print 'Loaded dataset `{:s}` for training'.format(dataset.name)

    if args.flipped:
        print 'Appending horizontally-flipped training examples...'
        dataset.append_flipped_images()

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(dataset)
    print 'Loaded {:d} training examples'.format(len(dataset))

    mkdirp(args.output)
    print 'Output will be saved to `{:s}`'.format(args.output)

    solver_file = '{:s}/solver.prototxt'.format(args.model)
    weights_file = '{:s}/weights.caffemodel'.format(args.model)

    solver = FasterRCNNSolver(
            solver_file=solver_file,
            weights_file=weights_file,
            dataset=dataset,
            output_dir=args.output,
            snapshot_iterations=args.snapshot_iterations)

    print 'Solving...'
    model_paths = solver.train_model(args.iterations)

    print 'done solving'
    sys.exit(0)
