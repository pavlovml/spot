# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from caffe.proto import caffe_pb2
from spot.utils.timer import Timer
import caffe
import google.protobuf as pb2
import numpy as np
import os
import spot.roi_data_layer.roidb as rdl_roidb

class FasterRCNNSolver(object):
    """
    A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, dataset, output_dir, solver_file,
            weights_file=None,
            snapshot_iterations=10000):
        self.output_dir = output_dir
        self.snapshot_iterations = snapshot_iterations

        self.load_solver(solver_file)

        if weights_file is not None:
            self.load_weights(weights_file)

        self.load_dataset(dataset)

    def load_solver(self, solver_file):
        self.solver = caffe.SGDSolver(solver_file)
        params = caffe_pb2.SolverParameter()
        with open(solver_file, 'rt') as f:
            pb2.text_format.Merge(f.read(), params)
        self.display = params.display
        self.snapshot_prefix = params.snapshot_prefix

    def load_weights(self, weights_file):
        print 'Loading model weights from {:s}'.format(weights_file)
        self.solver.net.copy_from(weights_file)

    def load_dataset(self, dataset):
        self.dataset = dataset

        print 'Computing bounding-box regression targets...'
        self.bbox_means, self.bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(dataset.roidb)

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

        filename = (self.snapshot_prefix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)
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
            if self.solver.iter % (10 * self.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % self.snapshot_iterations == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths
