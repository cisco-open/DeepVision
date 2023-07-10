ARG PYTORCH="1.13.0"
ARG CUDA="11.6"
ARG CUDNN="8"
ARG MM="2.8.0"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 wget git build-essential ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ENV FORCE_CUDA="1"

# Add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create the environment:
RUN conda create -n vqpy python=3.8.10

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "vqpy", "/bin/bash", "-c"]

# Install the necessary packages
COPY ./videoquery/vqpy/requirements.txt .
RUN pip install -r requirements.txt

# Install VQPy from Github
RUN pip install 'vqpy @ git+https://github.com/vqpy/vqpy.git'

# Deep Vision VQPy
RUN pip install redis
COPY ./videoquery/ /videoquery
WORKDIR /videoquery
