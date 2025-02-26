# sudo docker build -t helix .

# Base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV PATH="/usr/local/cuda-12.6/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}"

# Install basic tools and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libssl-dev \
    git \
    sudo \
    software-properties-common \
    python-is-python3 \
    python3-pip \
    pybind11-dev && \
    apt-get clean

# Install CMake (specific version)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3.tar.gz && \
    tar -xvzf cmake-3.28.3.tar.gz && \
    cd cmake-3.28.3 && \
    ./bootstrap && make -j$(nproc) && make install && \
    cd .. && rm -rf cmake-3.28.3 cmake-3.28.3.tar.gz

# Create user 'kth' (optional)
# RUN useradd -m -s /bin/bash kth && \
#     echo "kth ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /home/kth

# Install libzmq from source
RUN git clone https://github.com/zeromq/libzmq.git && \
    cd libzmq && \
    mkdir build && cd build && \
    cmake .. && make -j4 && make install && ldconfig

# Install cppzmq (header-only)
RUN git clone https://github.com/zeromq/cppzmq.git && \
    cd cppzmq && mkdir build && cd build && \
    cmake .. && make -j4 && make install

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /home/kth/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/home/kth/miniconda3/bin:$PATH"

# Create Conda environment with Python 3.10
RUN conda create -n runtime python=3.10 -y

# Install Python dependencies using conda run
RUN conda run -n runtime pip install --upgrade pip setuptools && \
    conda run -n runtime pip install vllm==0.4.0.post1 numpy~=1.26 && \
    conda run -n runtime conda install -c conda-forge libstdcxx-ng -y
    
# Copy Helix repository into container
COPY llm_sys/comm /home/kth/helix/llm_sys_comm
COPY docker_init.sh /home/kth/helix/docker_init.sh
COPY readme.md /home/kth/helix/readme.md
COPY setup.py /home/kth/helix/setup.py

# Install Helix's communication framework
WORKDIR /home/kth/helix/llm_sys_comm
# RUN ls -la /home/kth/helix/llm_sys/comm && sleep 30
RUN chmod +x build.sh

# Build the communication framework
RUN ls -la /home/kth/helix/llm_sys_comm && \
    conda run -n runtime bash -x build.sh

COPY examples /home/kth/helix/examples

# activate Conda environment
RUN conda init bash && \
    echo "conda activate runtime" >> ~/.bashrc

WORKDIR /home/kth/helix

RUN pip install \
    "numpy~=1.26" \
    "networkx~=3.2.1" \
    "matplotlib~=3.8.2" \
    "gurobipy~=11.0.0"

RUN pip install -e .