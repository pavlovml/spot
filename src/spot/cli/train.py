#! /usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from caffe.proto import caffe_pb2
from spot.config import setup_caffe
from spot.dataset import FasterRCNNDataset
from spot.utils.fs import mkdirp
from spot.utils.timer import Timer
import google.protobuf as pb2
import numpy as np
import os, caffe, sys, spot.net_factories

def save_prototxt(filename, proto):
    with open(filename, 'w+') as f:
        f.write(pb2.text_format.MessageToString(proto))

def save_labels(filename, labels):
    with open(filename, 'w+') as f:
        f.write('\n'.join(labels))

def create_solver_params(train_filename, iteration_size, lr_config):
    solver_params = caffe_pb2.SolverParameter()
    solver_params.train_net = train_filename
    solver_params.display = 1
    solver_params.average_loss = 100
    solver_params.iter_size = iteration_size
    solver_params.base_lr = lr_config['base']
    solver_params.lr_policy = lr_config['policy']
    solver_params.gamma = lr_config['gamma']
    solver_params.stepsize = lr_config['step_size']
    solver_params.momentum = lr_config['momentum']
    solver_params.weight_decay = lr_config['weight_decay']
    solver_params.snapshot = 0 # disable standard Caffe snapshots
    return solver_params

class Solver(object):
    """
    A simple wrapper around Caffe's solver.
    This wrapper gives us control over the snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(
            self,
            solver_filename,
            dataset,
            output_dir,
            weights_filename=None,
            snapshot_every=5000):
        self.output_dir = output_dir
        self.snapshot_every = snapshot_every
        self.solver = caffe.SGDSolver(solver_filename)

        if weights_filename is not None:
            print 'Loading model weights from {:s}'.format(weights_filename)
            self.solver.net.copy_from(weights_filename)

        print 'Computing bounding-box regression targets...'
        self.bbox_means, self.bbox_stds = dataset.add_bbox_regression_targets()

        self.solver.net.layers[0].set_roidb(dataset.roidb)

    def snapshot(self):
        """
        Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        # save original values
        orig_0 = net.params['bbox_pred'][0].data.copy()
        orig_1 = net.params['bbox_pred'][1].data.copy()

        # scale and shift with bbox reg unnormalization; then save snapshot
        net.params['bbox_pred'][0].data[...] = \
                (net.params['bbox_pred'][0].data *
                    self.bbox_stds[:, np.newaxis])
        net.params['bbox_pred'][1].data[...] = \
                (net.params['bbox_pred'][1].data *
                    self.bbox_stds + self.bbox_means)

        filename = os.path.join(
                self.output_dir,
                '{:d}.caffemodel'.format(self.solver.iter))

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        # restore net to original state
        net.params['bbox_pred'][0].data[...] = orig_0
        net.params['bbox_pred'][1].data[...] = orig_1

        return filename

    def train_model(self, iterations):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []

        try:
            while self.solver.iter < iterations:
                # Make one SGD update
                timer.tic()
                self.solver.step(1)
                timer.toc()
                if self.solver.iter % 10 == 0:
                    print 'speed: {:.3f}s / iter'.format(timer.average_time)

                if self.solver.iter % self.snapshot_every == 0:
                    last_snapshot_iter = self.solver.iter
                    model_paths.append(self.snapshot())

            if last_snapshot_iter != self.solver.iter:
                model_paths.append(self.snapshot())
        except KeyboardInterrupt:
            model_paths.append(self.snapshot())

        return model_paths

def add_subparser(parent):
    parser = parent.add_parser('train', help='Train a Faster R-CNN network')
    parser.set_defaults(func=run)

    parser.add_argument(
            'net_factory', metavar='NET_FACTORY',
            help='net factory',
            type=str)

    parser.add_argument(
            'dataset', metavar='DATASET',
            help='path to training dataset',
            type=str)

    parser.add_argument(
            'output', metavar='OUTPUT',
            help='directory to save snapshots to',
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

    training_group.add_argument(
            '-e', '--snapshot-every', dest='snapshot_every',
            help='number of iterations between snapshots [5000]',
            default=5000, type=int)

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

def run(args):
    setup_caffe(gpu=args.gpu, seed=args.seed)

    dataset = FasterRCNNDataset(
            args.dataset,
            include_flipped=args.flipped,
            enrich=True)

    print 'Loaded dataset `{:s}` for training ({:d} examples)' \
            .format(dataset.path, len(dataset.tags))

    mkdirp(args.output)
    print 'Output will be saved to `{:s}`'.format(args.output)

    net_factory = getattr(spot.net_factories, args.net_factory)

    labels_filename = os.path.join(args.output, 'labels.txt')
    train_filename = os.path.join(args.output, 'train.prototxt')
    test_filename = os.path.join(args.output, 'test.prototxt')
    solver_filename = os.path.join(args.output, 'solver.prototxt')

    train_params = net_factory(
       phase=caffe.TRAIN,
       num_classes=dataset.num_classes)

    test_params = net_factory(
       phase=caffe.TEST,
       num_classes=dataset.num_classes)

    solver_params = create_solver_params(
            train_filename=train_filename,
            iteration_size=args.iteration_size,
            lr_config={
                'base': args.lr_base,
                'policy': args.lr_policy,
                'gamma': args.lr_gamma,
                'step_size': args.lr_step_size,
                'momentum': args.lr_momentum,
                'weight_decay': args.lr_weight_decay
            })

    save_labels(labels_filename, dataset.indices_to_labels)
    save_prototxt(train_filename, train_params)
    save_prototxt(test_filename, test_params)
    save_prototxt(solver_filename, solver_params)

    solver = Solver(
            solver_filename=solver_filename,
            dataset=dataset,
            output_dir=args.output,
            weights_filename=args.weights.name if args.weights else None,
            snapshot_every=args.snapshot_every)

    print 'Solving...'
    model_paths = solver.train_model(args.iterations)

    print 'done solving'
    sys.exit(0)
