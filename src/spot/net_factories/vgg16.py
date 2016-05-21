from caffe import layers as L, params as P
import caffe

def max_pool(bottom, kernel_size=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=kernel_size, stride=stride)

def conv_relu(
        bottom, num_output,
        pad=1, kernel_size=3,
        param=[dict(lr_mult=0.0, decay_mult=0.0), dict(lr_mult=0.0, decay_mult=0.0)]):
    conv = L.Convolution(bottom, num_output=num_output, pad=pad, kernel_size=kernel_size, param=param)
    relu = L.ReLU(conv, in_place=True)
    return conv, relu

def fc_relu(bottom, param, num_output=4096):
    fc = L.InnerProduct(bottom, param=param, num_output=num_output)
    relu = L.ReLU(fc, in_place=True)
    return fc, relu

def create_net(phase, num_classes):
    n = caffe.NetSpec()

    if phase == caffe.TRAIN:
        param = [dict(lr_mult=1.0), dict(lr_mult=2.0)]
        n.data, n.im_info, n.gt_boxes = L.Python(
                name='input-data', ntop=3,
                module='spot.layers',
                layer='RoIDataLayer',
                param_str="'num_classes': {:d}".format(num_classes))
    else:
        param = [dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=0.0)]
        n.data, n.im_info = L.Data(ntop=2)

    # ==============================================================================
    # VGG16

    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64)
    n.conv1_2, n.relu1_2 = conv_relu(n.conv1_1, 64)
    n.pool1 = max_pool(n.conv1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.conv2_1, 128)
    n.pool2 = max_pool(n.conv2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, param=param)
    n.conv3_2, n.relu3_2 = conv_relu(n.conv3_1, 256, param=param)
    n.conv3_3, n.relu3_3 = conv_relu(n.conv3_2, 256, param=param)
    n.pool3 = max_pool(n.conv3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, param=param)
    n.conv4_2, n.relu4_2 = conv_relu(n.conv4_1, 512, param=param)
    n.conv4_3, n.relu4_3 = conv_relu(n.conv4_2, 512, param=param)
    n.pool4 = max_pool(n.conv4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, param=param)
    n.conv5_2, n.relu5_2 = conv_relu(n.conv5_1, 512, param=param)
    n.conv5_3, n.relu5_3 = conv_relu(n.conv5_2, 512, param=param)

    # ==============================================================================
    # RPN

    setattr(n, 'rpn/output', L.Convolution(
            n.conv5_3, name='rpn_conv/3x3',
            num_output=512, pad=1, kernel_size=3, stride=1,
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0.0),
            param=param))
    setattr(n, 'rpn_relu/3x3', L.ReLU(getattr(n, 'rpn/output'), in_place=True))

    n.rpn_cls_score = L.Convolution(
            getattr(n, 'rpn/output'),
            num_output=24, pad=0, kernel_size=1, stride=1,
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0.0),
            param=param)

    n.rpn_bbox_pred = L.Convolution(
            getattr(n, 'rpn/output'),
            num_output=48, pad=0, kernel_size=1, stride=1,
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0.0),
            param=param)

    n.rpn_cls_score_reshape = L.Reshape(
            n.rpn_cls_score,
            shape=dict(dim=[0, 2, -1, 0]))

    if phase == caffe.TRAIN:
        n.rpn_labels, n.rpn_bbox_targets, n.rpn_bbox_inside_weights, n.rpn_bbox_outside_weights = L.Python(
                n.rpn_cls_score, n.gt_boxes, n.im_info, n.data,
                name='rpn-data', ntop=4,
                module='spot.layers',
                layer='AnchorTargetLayer',
                param_str="'feat_stride': 16 \n'scales': !!python/tuple [4, 8, 16, 32]")

        n.rpn_cls_loss = L.SoftmaxWithLoss(
                n.rpn_cls_score_reshape, n.rpn_labels,
                name='rpn_loss_cls',
                propagate_down=[True, False],
                loss_weight=1.0,
                loss_param=dict(ignore_label=-1, normalize=True))

        n.rpn_loss_bbox = L.SmoothL1Loss(
                n.rpn_bbox_pred, n.rpn_bbox_targets, n.rpn_bbox_inside_weights, n.rpn_bbox_outside_weights,
                loss_weight=1.0,
                sigma=3.0)

    # ==============================================================================
    # RoI proposal

    n.rpn_cls_prob = L.Softmax(n.rpn_cls_score_reshape)

    n.rpn_cls_prob_reshape = L.Reshape(
            n.rpn_cls_prob,
            shape=dict(dim=[0, 24, -1, 0]))

    if phase == caffe.TRAIN:
        n.rpn_rois = L.Python(
                n.rpn_cls_prob_reshape, n.rpn_bbox_pred, n.im_info,
                name='proposal',
                module='spot.layers',
                layer='ProposalLayer',
                param_str="'feat_stride': 16 \n'scales': !!python/tuple [4, 8, 16, 32]")

        n.rois, n.labels, n.bbox_targets, n.bbox_inside_weights, n.bbox_outside_weights = L.Python(
                n.rpn_rois, n.gt_boxes,
                name='roi-data', ntop=5,
                module='spot.layers',
                layer='ProposalTargetLayer',
                param_str="'num_classes': {:d}".format(num_classes))
    else:
        n.rois = L.Python(
                n.rpn_cls_prob_reshape, n.rpn_bbox_pred, n.im_info,
                name='proposal',
                module='spot.layers',
                layer='ProposalLayer',
                param_str="'feat_stride': 16 \n'scales': !!python/tuple [4, 8, 16, 32]")

    # ==============================================================================
    # RCNN

    n.pool5 = L.ROIPooling(
            n.conv5_3, n.rois,
            name='roi_pool5',
            pooled_w=7, pooled_h=7, spatial_scale=0.0625)

    n.fc6, n.relu6 = fc_relu(n.pool5, param=param)
    n.fc7, n.relu7 = fc_relu(n.fc6, param=param)

    n.cls_score = L.InnerProduct(
            n.fc7,
            num_output=num_classes,
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0.0),
            param=param)

    n.bbox_pred = L.InnerProduct(
            n.fc7,
            num_output=4 * num_classes,
            weight_filler=dict(type='gaussian', std=0.001),
            bias_filler=dict(type='constant', value=0.0),
            param=param)

    if phase == caffe.TRAIN:
        n.loss_cls = L.SoftmaxWithLoss(
                n.cls_score, n.labels,
                name='loss_cls',
                propagate_down=[True, False],
                loss_weight=1.0)

        n.loss_bbox = L.SmoothL1Loss(
                n.bbox_pred, n.bbox_targets, n.bbox_inside_weights, n.bbox_outside_weights,
                loss_weight=1.0)
    else:
        n.cls_prob = L.Softmax(n.cls_score)

    proto = n.to_proto()
    proto.name = 'VGG_ILSVRC_16_layers'

    if phase == caffe.TEST:
        del proto.layer[0]
        proto.input.append('data')
        proto.input_shape.add().dim.extend([1, 3, 224, 224])
        proto.input.append('im_info')
        proto.input_shape.add().dim.extend([1, 3])

    return proto
