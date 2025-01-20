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

# Install Python dependencies using conda run
RUN pip install --upgrade pip setuptools && \
    pip install vllm==0.4.0.post1 numpy~=1.26 networkx~=3.2.1 matplotlib~=3.8.2 gurobipy~=11.0.0
    
# Copy Helix repository into container
COPY llm_sys/comm /home/kth/helix/llm_sys_comm
COPY docker_init.sh /home/kth/helix/docker_init.sh
COPY readme.md /home/kth/helix/readme.md
COPY setup.py /home/kth/helix/setup.py
COPY examples /home/kth/helix/examples
COPY env_export.sh /home/kth/helix/env_export.sh

# Install Helix's communication framework
WORKDIR /home/kth/helix/llm_sys_comm
RUN chmod +x build.sh

# Build the communication framework
RUN ls -la /home/kth/helix/llm_sys_comm && \
    ./build.sh

WORKDIR /home/kth/helix

RUN pip install pandas

RUN pip install -e .

RUN apt-get install -y \
    vim \
    net-tools \
    iputils-ping \
    iptables

WORKDIR /home/kth/helix/examples/real_sys

CMD ["bash"]