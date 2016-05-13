FROM pavlov/gibraltar:faster-rcnn
MAINTAINER Alex Kern <alex@pavlovml.com>

# install
RUN mkdir -p /spot
WORKDIR /spot
COPY . .
RUN cd /spot/src && \
    python setup.py install && \
    pip install -r requirements.txt && \
    cd /spot && rm -rf src/spot/build

# run
CMD bash
