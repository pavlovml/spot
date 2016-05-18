#! /usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from spot.fast_rcnn.train import FasterRCNNSolver
from spot.fast_rcnn.config import cfg
from spot.utils.fs import mkdirp
from spot.dataset import FasterRCNNDataset
import caffe, sys
import numpy as np
import spot.roi_data_layer.roidb as rdl_roidb

def add_subparser(parent):
    parser = parent.add_parser('train', help='Train a Faster R-CNN network')
    parser.set_defaults(func=run)

    parser.add_argument(
            'model', metavar='MODEL',
            help='model spec with train settings',
            type=file)

    parser.add_argument(
            'dataset', metavar='DATASET',
            help='path to training dataset',
            type=str)

    training_group = parser.add_argument_group('training arguments')

    training_group.add_argument(
            '-g', '--gpu', dest='gpu',
            help='GPU device id to use [0]',
            default=0, type=int)

    training_group.add_argument(
            '-w', '--weights', dest='weights',
            help='weights to fine-tune from',
            type=file)

    training_group.add_argument(
            '-i', '--iterations', dest='iterations',
            help='number of iterations to train [40000]',
            default=40000, type=int)

    training_group.add_argument(
            '-k', '--iteration-size', dest='iteration_size',
            help='number of images in each iteration [2]',
            default=2, type=int)

    training_group.add_argument(
            '-f', '--flipped', dest='flipped',
            help='include flipped images in training dataset',
            action='store_true')

    training_group.add_argument(
            '-s', '--seed', dest='seed',
            help='fixed RNG seed',
            default=None, type=int)

    snapshot_group = parser.add_argument_group('snapshot arguments')

    snapshot_group.add_argument(
            '-o', '--snapshot-dir', dest='snapshot_dir',
            help='directory to save snapshots to [output]',
            default='output', type=str)

    snapshot_group.add_argument(
            '-e', '--snapshot-every', dest='snapshot_every',
            help='number of iterations between snapshots [5000]',
            default=5000, type=int)

    snapshot_group.add_argument(
            '-p', '--snapshot-prefix', dest='snapshot_prefix',
            help='prefix for saved weights snapshots [spot]',
            default='spot', type=str)

    lr_group = parser.add_argument_group('learning rate arguments')

    lr_group.add_argument(
            '--lr-base', dest='lr_base',
            help='initial learning rate [0.001]',
            default=0.001, type=float)

    lr_group.add_argument(
            '--lr-policy', dest='lr_policy',
            help='learning rate policy [step]',
            default='step', type=str)

    lr_group.add_argument(
            '--lr-gamma', dest='lr_gamma',
            help='learning rate gamma [0.01]',
            default=0.01, type=float)

    lr_group.add_argument(
            '--lr-step-size', dest='lr_step_size',
            help='learning rate gamma [10000]',
            default=10000, type=int)

    lr_group.add_argument(
            '--lr-momentum', dest='lr_momentum',
            help='learning rate gamma [0.9]',
            default=0.9, type=float)

    lr_group.add_argument(
            '--lr-weight-decay', dest='lr_weight_decay',
            help='learning rate weight decay [0.0005]',
            default=0.0005, type=float)

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

    dataset = FasterRCNNDataset(
            args.dataset,
            include_flipped=args.flipped,
            enrich=True)

    print 'Loaded dataset `{:s}` for training ({:d} examples)'.format(dataset.name, len(dataset.tags))

    mkdirp(args.snapshot_dir)
    print 'Snapshots will be saved to `{:s}`'.format(args.snapshot_dir)

    lr_config = {
        'base': args.lr_base,
        'policy': args.lr_policy,
        'gamma': args.lr_gamma,
        'step_size': args.lr_step_size,
        'momentum': args.lr_momentum,
        'weight_decay': args.lr_weight_decay
    }

    solver = FasterRCNNSolver(
            model_file=args.model.name,
            weights_file=args.weights.name if args.weights else None,
            dataset=dataset,
            lr_config=lr_config,
            iteration_size=args.iteration_size,
            snapshot_dir=args.snapshot_dir,
            snapshot_every=args.snapshot_every,
            snapshot_prefix=args.snapshot_prefix)

    print 'Solving...'
    model_paths = solver.train_model(args.iterations)

    print 'done solving'
    sys.exit(0)
