# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from spot.fast_rcnn.config import cfg
from spot.utils.cython_bbox import bbox_overlaps
import PIL, fnmatch, json, os
import numpy as np
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

class SpotError(Exception):
    def __init__(self, message, cause=None):
        super(SpotError, self).__init__(message + (u', caused by ' + repr(cause) if cause else ''))
        self.cause = cause

# A roidb is a list of dictionaries, each with the following keys:
#   boxes
#   gt_overlaps
#   gt_classes
#   flipped
class FasterRCNNDataset(object):
    def __init__(self, path, ann_ext='json', image_ext='jpg'):
        self.path = path
        self._roidb = None
        self.annotation_files = glob_keyed_files(path, ann_ext)
        self.image_files = glob_keyed_files(path, image_ext)
        self.examples = list(self.image_files.keys())
        labels = self.load_label_set_annotation()
        self.label_set = LabelSet(labels)
        self.roidb = [self.load_image_annotation(i) for i in self.examples]

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
    def num_images(self):
        return len(self.examples)

    @property
    def image_index(self):
        return self.examples

    def load_annotation(self, name):
        try:
            filename = self.annotation_files[name]
            with open(filename) as f:
                data = json.load(f)
            return data
        except KeyError:
            raise SpotError("Could not find annotation '" + name + "'")
        except ValueError:
            raise SpotError("Could not decode JSON annotation '" + name + "'")

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

        return { 'boxes': boxes,
                 'gt_classes': gt_classes,
                 'gt_overlaps': overlaps,
                 'flipped': False }

    def image_path_at(self, i):
        """Return the absolute path to image i in the image sequence."""
        name = self.examples[i]
        return self.image_files[name]

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
            entry = { 'boxes': boxes,
                      'gt_overlaps': self.roidb[i]['gt_overlaps'],
                      'gt_classes': self.roidb[i]['gt_classes'],
                      'flipped': True }
            self.roidb.append(entry)
        self.examples = self.examples * 2
