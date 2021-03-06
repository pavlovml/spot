# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import json
import os
from datasets.imdb import IMDB
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import fnmatch
import cPickle
import subprocess

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

class OMF(IMDB):
    def __init__(self, path, ann_ext='json', image_ext='jpg'):
        IMDB.__init__(self, path)

        self.annotation_files = glob_keyed_files(path, ann_ext)
        self.image_files = glob_keyed_files(path, image_ext)
        self.examples = list(self.image_files.keys())

        labels = self.load_label_set_annotation()
        self.label_set = LabelSet(labels)

        self._roidb_handler = self.selective_search_roidb

        self.config = { 'cleanup': True, 'use_salt': True, 'top_k': 2000 }

    def __len__(self):
        return len(self.examples)

    def image_path_at(self, i):
        """Return the absolute path to image i in the image sequence."""
        name = self.examples[i]
        return self.image_files[name]

    def gt_roidb(self):
        return [self.load_image_annotation(i) for i in self.examples]

    def load_annotation(self, name):
        try:
            filename = self.annotation_files[name]
            with open(filename) as f:
                data = json.load(f)[0] # HACK
            return data
        except KeyError:
            raise OMFError("Could not find annotation '" + name + "'")
        except ValueError:
            raise OMFError("Could not decode JSON annotation '" + name + "'")

    def load_label_set_annotation(self, name='labels'):
        data = self.load_annotation(name)
        return data['labels']

    def load_image_annotation(self, name):
        data = self.load_annotation(name)
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

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
            print len(roidb)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['all_boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)])

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
