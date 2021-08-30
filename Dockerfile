FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
        python3-pip tmux \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache \
        matplotlib \
        pandas \
        scikit-learn \
        jupyterlab

RUN pip3 install --no-cache \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install pytorch-lightning

WORKDIR /work

CMD ["/bin/bash", "-c", "jupyter lab --no-browser --ip=* --allow-root"]
