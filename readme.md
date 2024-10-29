# ASPLOS'25 Helix
## Introduction 
Helix is a distributed system designed for high-throughput, low-latency large language model
serving across heterogeneous and potentially geo-distributed GPU clusters. This repository
contains the official implementation of both Helix's simulator and prototype system. Our paper
can be found here [https://arxiv.org/abs/2406.01566](https://arxiv.org/abs/2406.01566).

## Distributed LLM Serving Simulator Tutorial
The Helix simulator is a high-fidelity discrete-event simulator implemented in Python,
specifically designed for distributed LLM serving across heterogeneous and geo-distributed
GPU clusters. It provides detailed modeling and analysis of system behavior in complex
distributed environments.

### Installing Dependencies
We recommend using Python 3.11. To install the required dependencies, run the following command:
```bash
conda create -n helix python=3.11 -y && conda activate helix
pip install -e .
```
We use Gurobi as the MILP solver in the simulator, which requires a valid license. Please follow
the instructions on the [Gurobi website](https://www.gurobi.com/) to obtain a license.

### Running the Simulator
In this tutorial, we use an example of serving LLaMA-2 70B in a cluster with 24 machines to demonstrate
how to use Helix's simulator. The example is is located in `./examples/simulation`:
```bash
cd examples/simulation
```

### Step 1: Generate Configuration Files
First, we need to generate a cluster configuration file, which specifies the nodes and network connections
in the cluster. Run the following command to generate example cluster configuration files
```bash
python step1_gen_cluster.py
```
This Python script uses `FakeClusterGenerator` to generate a cluster with 24 machines located in a single
region (`config/single24.ini`) and `PartitionedClusterGenerator` to generate a cluster with 24 machines
located in three different regions (`config/3cluster24.ini`). For generation of other cluster topologies,
please refer to these two helper classes to implement your own generator. The simulator also needs a machine
profile file, which specifies the nic speed and vram size of machines. We provide such an example in
`config/machine_profile.ini`.

### Step 2: Finding Model Placement Plans

