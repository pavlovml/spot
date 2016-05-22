FROM pavlov/gibraltar:faster-rcnn
MAINTAINER Alex Kern <alex@pavlovml.com>

# install
RUN mkdir -p /spot
WORKDIR /spot
COPY . .
RUN cd /spot/src && \
    pip install -r requirements.txt && \
    python setup.py install && \
    cd /spot && rm -rf src

# run
CMD bash
