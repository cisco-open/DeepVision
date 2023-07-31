ARG PYTORCH="1.13.0"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && \
    apt-get install -y wget ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}

#COPY ./Ethosight/environment.yml /app/environment.yml
#
#COPY ./Ethosight/condacreate.sh /app/condacreate.sh

COPY ./Ethosight /app/


WORKDIR /app

RUN bash condacreate.sh

COPY requirements.txt .
COPY install_dependencies.sh .

RUN bash install_dependencies.sh

COPY . .
RUN rm -rf Ethosight

CMD ["tail", "-f", "/dev/null"]
