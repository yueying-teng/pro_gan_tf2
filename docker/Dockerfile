ARG img_digest=03e706e09b0425bb4f634a644c5d869f3b6d6c027411ccca14c18719121d3064
FROM tensorflow/tensorflow:2.4.1-gpu@sha256:${img_digest}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y  \
    git \
    wget \
    vim \
    pkg-config \
    build-essential \
    python-opencv

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /work
CMD ["bash"]

