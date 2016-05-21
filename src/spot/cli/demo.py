#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

from spot.config import cfg
from spot.test import im_detect
from spot.nms import nms
from spot.utils.timer import Timer
import caffe, os, sys, cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# TODO
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('output/{:s}.png'.format(class_name))

def demo(net, path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(path)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def add_subparser(parent):
    parser = parent.add_parser('demo', help='Demonstrate a Faster R-CNN network')
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
            '-g', '--gpu', dest='gpu',
            help='GPU device id to use',
            default=0, type=int)

    parser.add_argument(
            '-s', '--seed', dest='seed',
            help='fixed RNG seed',
            default=None, type=int)

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

    net = caffe.Net(args.model.name, args.weights.name, caffe.TEST)

    # Warmup on a dummy image
    print 'Evaluating detections'
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    paths = ['data/demo/000456.jpg', 'data/demo/000542.jpg', 'data/demo/001150.jpg',
                'data/demo/001763.jpg', 'data/demo/004545.jpg']
    for path in paths:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(path)
        demo(net, path)

    sys.exit(0)
