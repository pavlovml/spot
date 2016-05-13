# ==============================================================================
# config

.PHONY: default build run download-weights

default: install

# ==============================================================================
# phony targets

build:
	docker build -t pavlov/spotnet .

run: build
	docker run \
		-v nvidia_driver_361.42:/usr/local/nvidia \
		-v /home/ubuntu/data:/spot/data \
		-v /home/ubuntu/output:/spot/output \
		-v /home/ubuntu/models:/spot/models \
		--device /dev/nvidiactl:/dev/nvidiactl \
		--device /dev/nvidia-uvm:/dev/nvidia-uvm \
		--device /dev/nvidia0:/dev/nvidia0 \
		-it pavlov/spotnet

# time stdbuf -i0 -o0 -e0 spot train -w models/VGG16/weights.caffemodel models/VGG16/train.prototxt data/omf 2>&1 | tee -a output/train-`date +%Y-%m-%d-%H-%M-%S`.log

download-weights: models/imagenet_models.tgz
	cd models && tar zxfv imagenet_models.tgz
	mv models/imagenet_models/ZF.v2.caffemodel models/ZF/weights.caffemodel
	mv models/imagenet_models/VGG16.v2.caffemodel models/VGG16/weights.caffemodel
	mv models/imagenet_models/VGG_CNN_M_1024.v2.caffemodel models/VGG_CNN_M_1024/weights.caffemodel
	rmdir models/imagenet_models

# ==============================================================================
# file targets

models/imagenet_models.tgz:
	curl -o models/imagenet_models.tgz http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/imagenet_models.tgz
