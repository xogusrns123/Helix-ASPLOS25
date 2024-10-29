# ASPLOS'25 Helix
## Introduction 
Helix is a distributed system designed for high-throughput, low-latency large language model
serving across heterogeneous and potentially geo-distributed GPU clusters. This repository
contains the official implementation of both Helix's simulator and prototype system. Our paper
can be found here [https://arxiv.org/abs/2406.01566](https://arxiv.org/abs/2406.01566).

## Distributed LLM Serving Simulator
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
TODO

