FROM pavlov/gibraltar:faster-rcnn
MAINTAINER Alex Kern <alex@pavlovml.com>

# install
RUN mkdir -p /spot
WORKDIR /spot

ENV PATH=/spot/bin:$PATH PYTHONPATH=/spot/src:$PYTHONPATH
COPY . .
RUN cd /spot/src/spot && python setup.py build_ext --inplace && \
    cd /spot/src && pip install -r requirements.txt && \
    cd /spot && rm -rf src/spot/build

# run
CMD bash
