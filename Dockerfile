FROM pavlov/gibraltar:faster-rcnn
MAINTAINER Alex Kern <alex@pavlovml.com>

RUN mkdir /spot
WORKDIR /spot

COPY . .
RUN cd src && python setup.py build_ext --inplace
RUN rm -rf src/build
