# ==============================================================================
# config

.PHONY: default build build-lib build-caffe download-weights

default: install

MODEL ?= ZF

WEIGHTS ?= data/imagenet_models/$(MODEL).v2.caffemodel
SOLVER ?= models/$(MODEL)/solver.prototxt
ITERATIONS ?= 40000
DATASET ?= data/omf
	
# ==============================================================================
# phony targets

build: build-lib build-caffe

build-lib:
	cd lib && python setup.py build_ext --inplace
	rm -rf lib/build

build-caffe:
	git clone https://github.com/pavlovml/gibraltar.git
	cd gibraltar && $(MAKE)

download-weights: models/imagenet_models.tgz
	cd models && tar zxfv imagenet_models.tgz
	mv models/imagenet_models/ZF.v2.caffemodel models/ZF/weights.caffemodel
	mv models/imagenet_models/VGG16.v2.caffemodel models/VGG16/weights.caffemodel
	mv models/imagenet_models/VGG_CNN_M_1024.v2.caffemodel models/VGG_CNN_M_1024/weights.caffemodel
	rmdir models/imagenet_models

train:
	time ./tools/train_net.py \
		--gpu 0 \
		--solver $(SOLVER) \
		--weights $(WEIGHTS) \
		--imdb $(DATASET) \
		--iters $(ITERATIONS) \
		--cfg config/default.yml

# ==============================================================================
# file targets

models/imagenet_models.tgz:
	curl -o models/imagenet_models.tgz http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/imagenet_models.tgz
