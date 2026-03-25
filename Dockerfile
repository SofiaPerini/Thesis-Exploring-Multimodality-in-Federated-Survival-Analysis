# Start with a base Ubuntu image with CUDA and cuDNN pre-installed
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Install python3.10 and pip
ENV DEBIAN_FRONTEND noninteractive

# Set the working directory
WORKDIR /exp

# Create a new virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt


# Build and install DGL
RUN pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

ENV DGLBACKEND=pytorch
ENV DGL_HOME=/tmp/.dgl
ENV DGL_BUILD_GRAPHBOLT=0
ENV DGL_LOAD_GRAPHBOLT=0

# wandb key
ENV WANDB_API_KEY=5377528e74cd6e5c220d68f194d7d583e5e8e01b

# for running the notebooks
RUN pip install jupyter ipykernel ipywidgets
RUN mkdir -p /.local
RUN chmod -R 777 /.local

RUN mkdir -p /.cache
RUN chmod -R 777 /.cache


ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

USER root
CMD ["/bin/bash"]
USER root