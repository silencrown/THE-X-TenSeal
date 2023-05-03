FROM nvidia/cuda:11.0.3-base-ubuntu20.04

WORKDIR /home/THE-X

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget bzip2 ca-certificates curl git python3.8 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install THE-X-TenSeal
ENV PATH=/usr/local/bin:$PATH

RUN pip3 install --no-cache-dir torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html && \
    pip3 install --no-cache-dir transformers==4.11.3

# Install TenSEAL
RUN git clone https://github.com/silencrown/TenSEAL.git ../

RUN ../TenSEAL/.github/workflows/scripts/install_req_docker.sh

ENV CC=clang \
    CXX=clang++

RUN pip3 install ../TenSEAL

RUN git clone https://github.com/silencrown/THE-X-TenSeal .

RUN pip3 install --no-cache-dir -r ./THE-X-TenSeal/requirements.txt

LABEL org.opencontainers.image.authors="ShiqiGao"

CMD ["/bin/bash"]
