# ==============================================================================
# config

.PHONY: default build download-weights

default: install

DATASET ?= data/omf
GPU ?= 0
ITERATIONS ?= 40000
MODEL ?= ZF
NVIDIA_DRIVER ?= 361.42
OUTPUT_DIR ?= /home/ubuntu/output
MODELS_DIR ?= /home/ubuntu/models

# ==============================================================================
# phony targets

build:
	docker build -t pavlov/spot .

run: build
	docker run \
		-v nvidia_driver_$(NVIDIA_DRIVER):/usr/local/nvidia \
		-v /home/ubuntu/output:/spot/output \
		-v /home/ubuntu/models:/spot/models \
		--device /dev/nvidiactl:/dev/nvidiactl \
		--device /dev/nvidia-uvm:/dev/nvidia-uvm \
		--device /dev/nvidia0:/dev/nvidia0 \
		-it pavlov/spot

download-weights: models/imagenet_models.tgz
	cd models && tar zxfv imagenet_models.tgz
	mv models/imagenet_models/ZF.v2.caffemodel models/ZF/weights.caffemodel
	mv models/imagenet_models/VGG16.v2.caffemodel models/VGG16/weights.caffemodel
	mv models/imagenet_models/VGG_CNN_M_1024.v2.caffemodel models/VGG_CNN_M_1024/weights.caffemodel
	rmdir models/imagenet_models

train:
	./bin/spot-train \
		--gpu $(GPU) \
		--model $(MODEL) \
		--iterations $(ITERATIONS) \
		--config config/default.yml \
		$(DATASET)

# ==============================================================================
# file targets

models/imagenet_models.tgz:
	curl -o models/imagenet_models.tgz http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/imagenet_models.tgz
