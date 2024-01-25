FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

LABEL maintainer="f.madesta@uke.de"

ENV DEBIAN_FRONTEND=noninteractive
ARG ITK_VERSION=v5.3.0
ARG CMAKE_VERSION=3.26.4


RUN apt-get update -q && \
    apt-get install -y -q  \
        git \
        wget \
        openssl \
        libssl-dev \
        gengetopt \
        build-essential \
        zlib1g-dev \
        libopenmpi-dev

# install up-to-date cmake
WORKDIR /
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz
RUN tar -xf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && mv cmake-${CMAKE_VERSION}-linux-x86_64 /opt
RUN rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz
RUN ln -s /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/* /usr/local/bin

# clone ITK
RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git --branch ${ITK_VERSION}

# copy content (will be compiled later)
COPY docker /
