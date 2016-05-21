# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from spot.config import cfg
from spot.utils.bbox_transform import bbox_transform
from spot.utils.cython_bbox import bbox_overlaps
from spot.utils.fs import load_json_file, glob_keyed_files
import numpy as np
import random, scipy.sparse

def sample_from(l, n):
    extras = [random.choice(l) for _ in range(n - len(l))]
    result = l[:n] + extras
    random.shuffle(result)
    return result

def image_size(filename):
    import PIL
    return PIL.Image.open(filename).size

def value_to_index(l):
    return dict(zip(l, xrange(len(l))))

def _compute_targets(rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return np.zeros((rois.shape[0], 5), dtype=np.float32)
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
        np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets

def create_roidb(
        tags, labels_to_indices,
        include_flipped=False,
        enrich=False):
    roidb = []

    for tag in tags:
        label_index = labels_to_indices[tag['label']]

        boxes = np.zeros((1, 4), dtype=np.uint16)
        boxes[0, :] = tag['bounds']

        gt_classes = np.zeros((1), dtype=np.int32)
        gt_classes[0] = label_index

        overlaps = np.zeros((1, len(labels_to_indices)), dtype=np.float32)
        overlaps[0, label_index] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)

        entries = [{
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False
        }]

        if include_flipped or enrich:
            size = image_size(tag['image'])

            if include_flipped:
                flipped_boxes = boxes.copy()
                oldx1 = flipped_boxes[:, 0].copy()
                oldx2 = flipped_boxes[:, 2].copy()
                boxes[:, 0] = size[0] - oldx2
                boxes[:, 2] = size[0] - oldx1
                assert (flipped_boxes[:, 2] >= flipped_boxes[:, 0]).all()

                entries.append({
                    'boxes': flipped_boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'flipped': True
                })

            if enrich:
                for entry in entries:
                    entry['image'] = tag['image']
                    entry['width'] = size[0]
                    entry['height'] = size[1]
                    dense_overlaps = overlaps.toarray()
                    # max overlap with gt over classes (columns)
                    max_overlaps = dense_overlaps.max(axis=1)
                    # gt class that had the max overlap
                    max_classes = dense_overlaps.argmax(axis=1)
                    entry['max_classes'] = max_classes
                    entry['max_overlaps'] = max_overlaps
                    # sanity checks
                    # max overlap of 0 => class should be zero (background)
                    zero_inds = np.where(max_overlaps == 0)[0]
                    assert all(max_classes[zero_inds] == 0)
                    # max overlap > 0 => class should not be zero (must be a fg class)
                    nonzero_inds = np.where(max_overlaps > 0)[0]
                    assert all(max_classes[nonzero_inds] != 0)

        roidb += entries

    return roidb

class FasterRCNNDataset(object):
    def __init__(
            self, path,
            tag_ext='json',
            img_ext='jpg',
            include_flipped=False,
            enrich=False):
        self.path = path

        # Iterate over all of the JSON tag files, associating each with a
        # corresponding image file.
        count_by_label = {}
        tags_by_label = {}
        tag_files = glob_keyed_files(path, tag_ext)
        img_files = glob_keyed_files(path, img_ext)
        for name, tag_filename in tag_files.items():
            tags = load_json_file(tag_filename)

            # TODO: Support multiple tags per image.
            if len(tags) == 0:
                continue
            print tag_filename, repr(tags)
            tag = tags[0]

            label = tag['label']
            if label not in tags_by_label:
                count_by_label[label] = 0
                tags_by_label[label] = []

            count_by_label[label] += 1
            tags_by_label[label].append({
                'label': label,
                'image': img_files[name],
                'bounds': tag['bounds']
            })

        # Ensure that each label has an equal number of tags by sampling from
        # the provided tags.
        self.tags = []
        max_count_by_label = max(count_by_label.values())
        for label, tags in tags_by_label.items():
            self.tags += sample_from(tags, max_count_by_label)

        # Associate each label with an index based on its sort order.
        self.indices_to_labels = ['__background__'] + sorted(tags_by_label.keys())
        self.labels_to_indices = value_to_index(self.indices_to_labels)

        # Create the roidb for Faster-RCNN.
        self.roidb = create_roidb(
                self.tags, self.labels_to_indices,
                include_flipped=include_flipped,
                enrich=enrich)

        # TODO
        self.num_classes = len(self.indices_to_labels)
        self.num_images = len(self.roidb)

    def add_bbox_regression_targets(self):
        """Add information needed to train bounding-box regressors."""
        assert len(self.roidb) > 0
        assert 'max_classes' in self.roidb[0], 'Did you call prepare_roidb first?'

        # Infer number of classes from the number of columns in gt_overlaps
        for im_i in xrange(self.num_images):
            rois = self.roidb[im_i]['boxes']
            max_overlaps = self.roidb[im_i]['max_overlaps']
            max_classes = self.roidb[im_i]['max_classes']
            self.roidb[im_i]['bbox_targets'] = \
                    _compute_targets(rois, max_overlaps, max_classes)

        # Use fixed / precomputed "means" and "stds" instead of empirical values
        means = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self.num_classes, 1))
        stds = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self.num_classes, 1))

        print 'bbox target means:'
        print means
        print means[1:, :].mean(axis=0) # ignore bg class
        print 'bbox target stdevs:'
        print stds
        print stds[1:, :].mean(axis=0) # ignore bg class

        print "Normalizing targets"
        for im_i in xrange(self.num_images):
            targets = self.roidb[im_i]['bbox_targets']
            for cls in xrange(1, self.num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                self.roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                self.roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]

        # These values will be needed for making predictions
        # (the predicts will need to be unnormalized and uncentered)
        return means.ravel(), stds.ravel()
