ARG PYTORCH="1.13.0"
ARG CUDA="11.6"
ARG CUDNN="8"
ARG MM="2.8.0"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
RUN pip install mmdet==2.28.1

RUN conda clean --all
ENV FORCE_CUDA="1"

#RUN git clone https://github.com/open-mmlab/mmtracking.git /mmtracking
COPY . /mmtracking
WORKDIR /mmtracking
RUN pip install -r requirements.txt
RUN pip install -r mmtracking/requirements/build.txt
RUN pip install -v -e mmtracking/
