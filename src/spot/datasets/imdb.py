# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from spot.fast_rcnn.config import cfg
from spot.utils.cython_bbox import bbox_overlaps
import PIL
import fnmatch
import json
import numpy as np
import os
import os.path as osp
import scipy.sparse

def glob_keyed_files(path, ext):
    res = {}
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.' + ext):
            key = os.path.splitext(os.path.basename(filename))[0]
            res[key] = os.path.join(root, filename)
    return res

class LabelSet:
    def __init__(self, labels):
        self.labels = labels
        self.indices = dict(zip(labels, xrange(len(labels))))

    def __len__(self):
        return len(self.labels)

    def index(self, label):
        return self.indices[label]

    def label(self, index):
        return self.labels[index]

class OMFError(Exception):
    def __init__(self, message, cause=None):
        super(OMFError, self).__init__(message + (u', caused by ' + repr(cause) if cause else ''))
        self.cause = cause

class IMDB(object):
    """Image database."""

    def __init__(self, path, ann_ext='json', image_ext='jpg'):
        self.path = path
        self._roidb = None
        self.annotation_files = glob_keyed_files(path, ann_ext)
        self.image_files = glob_keyed_files(path, image_ext)
        self.examples = list(self.image_files.keys())
        labels = self.load_label_set_annotation()
        self.label_set = LabelSet(labels)

    def __len__(self):
        return len(self.examples)

    @property
    def name(self):
        return self.path

    @property
    def num_classes(self):
        return len(self.label_set)

    @property
    def classes(self):
        return self.label_set

    @property
    def image_index(self):
        return self.examples

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = [self.load_image_annotation(i) for i in self.examples]
        return self._roidb

    def load_annotation(self, name):
        try:
            filename = self.annotation_files[name]
            with open(filename) as f:
                data = json.load(f)
            return data
        except KeyError:
            raise OMFError("Could not find annotation '" + name + "'")
        except ValueError:
            raise OMFError("Could not decode JSON annotation '" + name + "'")

    def load_label_set_annotation(self, name='labels'):
        data = self.load_annotation(name)
        return data['labels']

    def load_image_annotation(self, name):
        data = self.load_annotation(name)[0]
        label_index = self.label_set.index(data['label'])

        boxes = np.zeros((1, 4), dtype=np.uint16)
        boxes[0, :] = data['bounds']

        gt_classes = np.zeros((1), dtype=np.int32)
        gt_classes[0] = label_index

        overlaps = np.zeros((1, len(self.label_set)), dtype=np.float32)
        overlaps[0, label_index] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)

        print boxes, gt_classes
        return { 'boxes': boxes,
                 'gt_classes': gt_classes,
                 'gt_overlaps': overlaps,
                 'flipped': False }

    @property
    def num_images(self):
        return len(self.examples)

    def image_path_at(self, i):
        """Return the absolute path to image i in the image sequence."""
        name = self.examples[i]
        return self.image_files[name]

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def append_flipped_images(self):
        num_images = self.num_images
        widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(self.num_images)]
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 # - 1
            boxes[:, 2] = widths[i] - oldx1 # - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self.examples = self.examples * 2

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        """Evaluate detection proposal recall metrics.

        Returns:
            results: dictionary of results with keys
                'ar': average recall
                'recalls': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = { 'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                  '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [ [0**2, 1e5**2],    # all
                        [0**2, 32**2],     # small
                        [32**2, 96**2],    # medium
                        [96**2, 1e5**2],   # large
                        [96**2, 128**2],   # 96-128
                        [128**2, 256**2],  # 128-256
                        [256**2, 512**2],  # 256-512
                        [512**2, 1e5**2],  # 512-inf
                      ]
        assert areas.has_key(area), 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in xrange(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in xrange(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert(gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert(_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps}
