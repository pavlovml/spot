# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from caffe.proto import caffe_pb2
from spot.utils.timer import Timer
import os, tempfile, caffe
import google.protobuf as pb2
import numpy as np
import spot.roi_data_layer.roidb as rdl_roidb

DEFAULT_LR_CONFIG = {
    'base': 0.001,
    'policy': 'step',
    'gamma': 0.01,
    'step_size': 10000,
    'momentum': 0.9,
    'weight_decay': 0.0005
}

def save_prototxt(proto):
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(pb2.text_format.MessageToString(proto))
    f.close()
    return f.name

class FasterRCNNSolver(object):
    """
    A simple wrapper around Caffe's solver.
    This wrapper gives us control over the snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(
            self, net_factory, dataset,
            weights_file=None,
            lr_config=DEFAULT_LR_CONFIG,
            iteration_size=2,
            snapshot_dir='output',
            snapshot_every=5000,
            snapshot_prefix='spot'):
        self.net_factory = net_factory
        self.dataset = dataset
        self.weights_file = weights_file
        self.lr_config = lr_config
        self.iteration_size = iteration_size
        self.snapshot_dir = snapshot_dir
        self.snapshot_every = snapshot_every
        self.snapshot_prefix = snapshot_prefix

        self.solver = self.create_solver()

        if weights_file is not None:
            print 'Loading model weights from {:s}'.format(weights_file)
            self.solver.net.copy_from(weights_file)

        print 'Computing bounding-box regression targets...'
        self.bbox_means, self.bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(dataset.roidb)

        self.solver.net.layers[0].set_roidb(dataset.roidb)

    def create_solver(self):
        net_params = self.net_factory(num_classes=self.dataset.num_classes)
        train_prototxt = save_prototxt(net_params)
        solver_params = caffe_pb2.SolverParameter()
        solver_params.train_net = train_prototxt
        solver_params.display = 1
        solver_params.average_loss = 100
        solver_params.iter_size = self.iteration_size
        solver_params.base_lr = self.lr_config['base']
        solver_params.lr_policy = self.lr_config['policy']
        solver_params.gamma = self.lr_config['gamma']
        solver_params.stepsize = self.lr_config['step_size']
        solver_params.momentum = self.lr_config['momentum']
        solver_params.weight_decay = self.lr_config['weight_decay']
        solver_params.snapshot = 0 # disable standard Caffe snapshots
        solver_prototxt = save_prototxt(solver_params)
        return caffe.SGDSolver(solver_prototxt)

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

        filename = (self.snapshot_prefix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.snapshot_dir, filename)
        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        # restore net to original state
        net.params['bbox_pred'][0].data[...] = orig_0
        net.params['bbox_pred'][1].data[...] = orig_1
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
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
        return model_paths
