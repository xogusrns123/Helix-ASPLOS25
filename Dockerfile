# sudo docker build -t helix .

# FROM nvidia/cuda:12.6.1-devel-ubuntu24.04
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV PATH="/usr/local/cuda-12.6/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}"

# Install basic tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    sudo \
    software-properties-common \
    python-is-python3 \
    python3-pip \
    pybind11-dev && \
    apt-get clean

# Create user 'kth' and set home directory
RUN useradd -m -s /bin/bash kth && \
    echo "kth ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER kth
WORKDIR /home/kth

# Install libzmq from source using cmake
RUN git clone https://github.com/zeromq/libzmq.git && \
    cd libzmq && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4 && \
    sudo make install && \
    sudo ldconfig

# Install cppzmq (header-only) using cmake
RUN git clone https://github.com/zeromq/cppzmq.git && \
    cd cppzmq && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4 && \
    sudo make install

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /home/kth/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/home/kth/miniconda3/bin:$PATH"

# Create and activate Conda environment
RUN conda init bash && \
    conda create -n runtime python=3.10 -y && \
    echo "conda activate runtime" >> ~/.bashrc

# Install Python dependencies in the Conda environment
RUN /bin/bash -c "source activate runtime && \
    pip install vllm==0.4.0.post1 numpy~=1.26 && \
    conda install -n runtime -c conda-forge libstdcxx-ng -y"


# Copy Helix repository into container
WORKDIR /home/kth/helix
COPY . .

# # Install Helix's communication framework
RUN /bin/bash -c "source activate runtime && sudo pip install -e ."

# # Build the communication framework
WORKDIR /home/kth/helix/llm_sys/comm
# # Ensure build.sh has execution permissions
RUN sudo chmod +x build.sh

# # # Debugging: Check directory structure and run the script
RUN ls -la /home/kth/helix/llm_sys/comm && sudo bash -x build.sh

# # # Verify installation by running the unit test
WORKDIR /home/kth/helix/llm_sys/comm/build
RUN ./test_msg