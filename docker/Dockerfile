FROM nvidia/cuda:11.0.3-base-ubuntu20.04

WORKDIR /home/THE-X

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget bzip2 ca-certificates curl git python3.8 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/bin:$PATH

RUN pip3 install --no-cache-dir torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html && \
    pip3 install --no-cache-dir transformers==4.11.3

# Install TenSEAL
RUN mkdir /home/TenSEAL && \
    git clone https://github.com/silencrown/TenSEAL.git /home/TenSEAL

RUN apt update -y && \
    apt install build-essential automake libtool libtool-bin clang -y

ENV CC=clang \
    CXX=clang++

RUN cmake_install="cmake-3.17.0-Linux-x86_64.sh" && \
    wget https://github.com/Kitware/CMake/releases/download/v3.17.0/${cmake_install} && \
    sh ${cmake_install} --skip-license --prefix=/usr/local/ --exclude-subdir

RUN python3 -m pip install --upgrade pip && \
    pip install -r /home/TenSEAL/requirements_dev.txt && \
    pip install setuptools wheel twine auditwheel

RUN pip3 install /home/TenSEAL

# Install THE-X-TenSeal
RUN git clone https://github.com/silencrown/THE-X-TenSeal .

RUN pip3 install --no-cache-dir -r ./THE-X-TenSeal/requirements.txt

LABEL org.opencontainers.image.authors="ShiqiGao"

CMD ["/bin/bash"]
