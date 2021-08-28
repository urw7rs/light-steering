FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
        python3-pip tmux \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache \
        matplotlib \
        pandas \
        scikit-learn \
        jupyterlab \
        torch \
        torchvision

WORKDIR /work

CMD ["/bin/bash", "-c", "jupyter lab --no-browser --ip=* --allow-root"]
