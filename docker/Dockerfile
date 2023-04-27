FROM nvidia/cuda:11.1-base-ubuntu20.04

WORKDIR /home/THE-X

RUN apt-get update && apt-get install -y wget bzip2 ca-certificates curl git python3.8 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/bin:$PATH

RUN pip3 install --no-cache-dir torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html && \
    pip3 install --no-cache-dir transformers==4.11.3

RUN git clone https://github.com/silencrown/THE-X-TenSeal .

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r ./THE-X-TenSeal/requirements.txt

CMD ["/bin/bash"]
