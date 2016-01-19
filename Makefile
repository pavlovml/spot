# ==============================================================================
# config

.PHONY: default build build-lib build-caffe download-models

default: install
	
# ==============================================================================
# phony targets

build: build-lib build-caffe

build-lib:
	cd lib && python setup.py build_ext --inplace
	rm -rf lib/build

build-caffe:
	git clone https://github.com/pavlovml/gibraltar.git
	cd gibraltar && $(MAKE)

download-models: data/faster_rcnn_models data/imagenet_models

# ==============================================================================
# file targets

data/faster_rcnn_models.tgz:
	curl -o data/faster_rcnn_models.tgz http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/faster_rcnn_models.tgz

data/faster_rcnn_models: | data/faster_rcnn_models.tgz
	cd data && tar zxfv faster_rcnn_models.tgz

data/imagenet_models.tgz:
	curl -o data/imagenet_models.tgz http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/imagenet_models.tgz

data/imagenet_models: | data/imagenet_models.tgz
	cd data && tar zxfv imagenet_models.tgz
