# Artifact Evaluation for Helix - Reproducibility

We assume that the environment has already been set up based on the `readme.md` in the
root directory of the repository. You can use the host machine in the cluster to run
simulator experiments. You will need the whole cluster of 24 machines to run the
experiments for the prototype system. Before starting artifact evaluation for
reproducibility, run the following command to activate the conda environment:

```bash
conda activate runtime
```

> **Note:** we provide a copy of the result files we get after running the commands.

## Section 6.3 Single Cluster

All files related to this group of experiments are located in `artifact_evaluation/single_cluster`.
Start from the root directory of the repository:
```bash
cd artifact_evaluation/single_cluster
```

### Step 1: Generate Cluster Config Files
Run the following command to generate cluster config files:
```bash
python step1_gen_cluster.py
```
It will automatically generate the config file that represents a cluster with 24 machines 
(4 1xA100 machines, 8 1xL4 machines, 12 1xT4 machines) in `config/cluster24.ini`. It will
also generate the three sub-cluster config files with each type of machine: `config/a100.ini`,
`config/l4.ini` and `config/t4.ini`.

### Model Placement

