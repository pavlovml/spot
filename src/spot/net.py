from spot.fast_rcnn.config import cfg
import caffe, imp
import numpy as np

def setup_caffe(gpu=0, seed=None):
    """Initializes Caffe's python bindings."""
    cfg.GPU_ID = gpu

    if seed:
        np.random.seed(seed)
        caffe.set_random_seed(seed)

    caffe.set_mode_gpu()
    caffe.set_device(gpu)

def load_train_net(path):
    net_module = imp.load_source('spot.net', path)
    return net_module.train_net

def load_test_net(path):
    net_module = imp.load_source('spot.net', path)
    return net_module.test_net
