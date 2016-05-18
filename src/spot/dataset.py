# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from spot.utils.fs import load_json_file, glob_keyed_files
import random
import numpy as np
import scipy.sparse

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
