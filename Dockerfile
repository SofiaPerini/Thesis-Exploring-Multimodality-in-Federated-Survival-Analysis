# Start with a base Ubuntu image with CUDA and cuDNN pre-installed
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Install python3.10 and pip
ENV DEBIAN_FRONTEND noninteractive

#RUN apt-get update && apt-get install -y \
#  build-essential ca-certificates python3.10 python3.10-dev python3.10-distutils git vim wget cmake python3-pip python3.10-venv
#RUN ln -sv /usr/bin/python3.10 /usr/bin/python
#RUN ln -svf /usr/bin/python3.10 /usr/bin/python3

# Install system dependencies
#RUN apt update && apt install -y \
#    build-essential \
#    curl \
#    software-properties-common \
#    git \
#    wget

# Set the working directory
WORKDIR /exp

# Create a new virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Enable OpenCV  (remove comment if needed)
#RUN apt install -y libsm6 libxext6 libxrender-dev ffmpeg

# Install Python dependencies
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

#RUN apt-get update && apt-get install -y \
#    git build-essential cmake libopenblas-dev libomp-dev

# Build and install DGL
RUN pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

ENV DGLBACKEND=pytorch
ENV DGL_HOME=/tmp/.dgl
ENV DGL_BUILD_GRAPHBOLT=0
ENV DGL_LOAD_GRAPHBOLT=0

# wandb key
ENV WANDB_API_KEY=5377528e74cd6e5c220d68f194d7d583e5e8e01b


# run of clean_datasets.py
#COPY ./src/clean_datasets.py  / /
#USER root
#RUN mkdir -p /src/datasets_csv/metadata && chmod -R 777 /src/datasets_csv/metadata
#RUN mkdir -p /src/datasets_csv/clinical_data && chmod -R 777 /src/datasets_csv/clinical_data
#RUN mkdir -p /src/datasets_csv/raw_rna_data/combine/brca && chmod -R 777 /src/datasets_csv/raw_rna_data/combine/brca

#CMD [ "python", "./clean_datasets.py"]

# run find_missing_ids
#COPY ./src/check_missing_wsi.py  / /
#USER root
#CMD [ "python", "./check_missing_wsi.py"]

#ENV SHELL /bin/bash

# for running the notebooks
RUN pip install jupyter ipykernel ipywidgets
RUN mkdir -p /.local
RUN chmod -R 777 /.local

RUN mkdir -p /.cache
RUN chmod -R 777 /.cache

#COPY src/fedavg_clientsv_mio.sh /expsurv_fedavg_clients.shi#.sh
#RUN chmod +x /expsurv_fedavg_clientsv_mi#.sh
#RUN chmod 777 /expsurv_fedavg_clientsv_mio.sh

#CMD ["src/surv_mio.sh"]

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USER root
CMD ["/bin/bash"]
USER root