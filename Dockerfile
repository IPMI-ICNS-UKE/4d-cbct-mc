FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

LABEL maintainer="f.madesta@uke.de"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q && \
    apt-get install -y -q  \
        git \
        zlib1g-dev \
        build-essential


# copy mcgpu (will be compiled later)
COPY mcgpu /mcgpu
