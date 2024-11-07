# Artifact Evaluation for Helix - Reproducibility

We assume that the environment has already been set up based on the `readme.md` in the
root directory of the repository. You can use the host machine in the cluster to run
simulator experiments. You will need the whole cluster of 24 machines to run the
experiments for the prototype system. A copy of the artifact is stored in `~/helix`. 
Before starting artifact evaluation for reproducibility, run the following command to
activate the conda environment:

```bash
conda activate runtime
```

> **Note:** For the experiments below, we provide a copy of the results we get after running
> them.

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

### Step 2: Model Placement

The next step is to generate the model placement for the cluster. Run the following commands
to generate model placements for LLaMA-1 30B and LLaMA-2 70B using different model placement
methods. Notice that before running Helix's MILP-based model placement planner, you need to
remove the `./layout_llama30b/ilp` and `./layout_llama70b/ilp` directories, which currently
contains the result we get. We suggest moving them to a backup place if you want to compare
your results with ours. Also, for `llama70b`, you need to run `petals` before running
Helix's MILP model placement planner, as we bootstrap the solver with `petals`' solution.

> **Notes:** We notice that Gurobi produces completely different optimization traces when
> using **different licenses, even when using the same random seed**. When using the default
> limited license, the optimization performance is much worse than that of using the academic
> license. (objective value = 952 v.s. 1289) Unfortunately, we are not allowed and unable to
> bind our academic Gurobi license to the cluster provided. Note that this is an issue with
> Gurobi **licensing** instead of our system, and we provide our optimization trace in
> `./layout_llama70b/ilp/trace.txt` for you to compare against."

> **Notes:** Running Helix to search for a model placement for LLaMA 70B may take a long time.
> We set the max running time to 10 hours, but you can stop the solver at any time with `ctrl +c`. 
> In our experiments, on a machine with 14 cores and academic license, we manually early-stop
> the solver at round 10 minutes. This solution at this point already has good quality. The
> objective value (Incumbent) equals to 1289.

```bash
# Generate model placement using heuristic method Petals
python step2_gen_layout.py petals llama30b
python step2_gen_layout.py petals llama70b
# Generate model placement using heuristic method Swarm
python step2_gen_layout.py swarm llama30b
python step2_gen_layout.py swarm llama70b
# Generate model placement using heuristic method separate pipelines
python step2_gen_layout.py separate llama30b
python step2_gen_layout.py separate llama70b
# Generate model placement using Helix's MILP-based method
python step2_gen_layout.py ilp llama30b  # remove ./layout_llama30b/ilp before running
python step2_gen_layout.py ilp llama70b  # remove ./layout_llama70b/ilp before running
```

After running the commands above, you will get model placement files located in:
+ **petals & llama 30b**: `./layout_llama30b/petals` (`petals_sol.ini` and `simulator_cluster.ini`)
+ **petals & llama 70b**: `./layout_llama70b/petals` (`petals_sol.ini` and `simulator_cluster.ini`)
+ **swarm & llama 30b**: `./layout_llama30b/swarm` (`swarm_sol.ini` and `simulator_cluster.ini`)
+ **swarm & llama 70b**: `./layout_llama70b/swarm` (`swarm_sol.ini` and `simulator_cluster.ini`)
+ **separate & llama 30b**: `./layout_llama30b/separate` (6 manually created files)
+ **separate & llama 70b**: `./layout_llama70b/separate` (6 manually created files)
+ **ilp & llama 30b**: `./layout_llama30b/ilp/a100`, `./layout_llama30b/ilp/l4`, `./layout_llama30b/ilp/t4`, each containing the model placement for a sub-cluster
+ **ilp & llama 70b**: `./layout_llama70b/ilp` (`ilp_sol.ini` and `simulator_cluster.ini`, and 3 other files that records information about the MILP problem)

### Step 3: Run Simulation
With the model placement files generated, we can run the simulation and reproduce the results in the
paper.

(1) Run LLaMA 30B in offline setup using Helix and observe its decode throughput. This
corresponds to Figure 5(a)'s Simulation - Helix in the paper.
```bash
python step3_simulation.py helix llama30b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B offline simulation results: Helix
Total decode throughput: 339.6 tokens/s
************************************************************
```

(2) Run LLaMA 30B in offline setup using Swarm and observe its decode throughput. This
corresponds to Figure 5(a)'s Simulation - Swarm in the paper.
```bash
python step3_simulation.py swarm llama30b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B offline simulation results: Swarm
Total decode throughput: 151.6 tokens/s
************************************************************
```

(3) Run LLaMA 30B in offline setup using Separate Pipelines and observe its decode 
throughput. This corresponds to Figure 5(a)'s Simulation - Separate Pipelines (SP) in the paper.
```bash
python step3_simulation.py separate llama30b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B offline simulation results: Separate
Total decode throughput: 309.0 tokens/s
************************************************************
```

(4) Run LLaMA 30B in online setup using Helix and observe its decode throughput, prompt latency
and decode latency. The decode throughput corresponds to Figure 5(b)'s Simulation - Helix; the
prompt latency corresponds to Figure 5(e)'s Simulation - Helix; the decode latency corresponds
to Figure 5(f)'s Simulation - Helix.
```bash
python step3_simulation.py helix llama30b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B online simulation results: Helix
Total decode throughput: 216.7 tokens/s
Prompt latency:
Latency 5th percentile: 0.10 s
Latency 25th percentile: 0.17 s
Latency 50th percentile: 0.43 s
Latency 75th percentile: 0.99 s
Latency 95th percentile: 2.95 s
Decode latency:
Latency 5th percentile: 0.06 s
Latency 25th percentile: 0.08 s
Latency 50th percentile: 0.13 s
Latency 75th percentile: 0.37 s
Latency 95th percentile: 0.70 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama30b/ilp_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(5) Run LLaMA 30B in online setup using Swarm and observe its decode throughput, prompt latency
and decode latency. The decode throughput corresponds to Figure 5(b)'s Simulation - Swarm; the
prompt latency corresponds to Figure 5(e)'s Simulation - Swarm; the decode latency corresponds
to Figure 5(f)'s Simulation - Swarm.
```bash
python step3_simulation.py swarm llama30b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B online simulation results: Swarm
Total decode throughput: 104.9 tokens/s
Prompt latency:
Latency 5th percentile: 0.42 s
Latency 25th percentile: 0.71 s
Latency 50th percentile: 1.42 s
Latency 75th percentile: 1.75 s
Latency 95th percentile: 2.28 s
Decode latency:
Latency 5th percentile: 0.22 s
Latency 25th percentile: 0.25 s
Latency 50th percentile: 0.26 s
Latency 75th percentile: 0.29 s
Latency 95th percentile: 0.47 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama30b/swarm_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(6) Run LLaMA 30B in online setup using Separate Pipelines and observe its decode throughput,
prompt latency and decode latency. The decode throughput corresponds to Figure 5(b)'s Simulation -
Separate Pipelines (SP); the prompt latency corresponds to Figure 5(e)'s Simulation - Separate
Pipelines (SP); the decode latency corresponds to Figure 5(f)'s Simulation - Separate Pipelines (SP).
```bash
python step3_simulation.py separate llama30b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B online simulation results: Separate
Total decode throughput: 189.8 tokens/s
Prompt latency:
Latency 5th percentile: 0.09 s
Latency 25th percentile: 0.16 s
Latency 50th percentile: 0.42 s
Latency 75th percentile: 1.07 s
Latency 95th percentile: 2.99 s
Decode latency:
Latency 5th percentile: 0.06 s
Latency 25th percentile: 0.08 s
Latency 50th percentile: 0.11 s
Latency 75th percentile: 0.33 s
Latency 95th percentile: 0.57 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama30b/separate_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(7) Run LLaMA 70B in offline setup using Helix and observe its decode throughput. This
corresponds to Figure 5(c)'s Simulation - Helix in the paper.
```bash
python step3_simulation.py helix llama70b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B offline simulation results: Helix
Total decode throughput: 252.1 tokens/s
************************************************************
```

(8) Run LLaMA 70B in offline setup using Swarm and observe its decode throughput. This
corresponds to Figure 5(c)'s Simulation - Swarm in the paper.
```bash
python step3_simulation.py swarm llama70b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B offline simulation results: Swarm
Total decode throughput: 124.7 tokens/s
************************************************************
```

(9) Run LLaMA 70B in offline setup using Separate Pipelines and observe its decode throughput.
This corresponds to Figure 5(c)'s Simulation - Separate Pipelines (SP) in the paper.
```bash
python step3_simulation.py separate llama70b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B offline simulation results: Separate
Total decode throughput: 124.3 tokens/s
************************************************************
```

(10) Run LLaMA 70B in online setup using Helix and observe its decode throughput, prompt latency
and decode latency. The decode throughput corresponds to Figure 5(d)'s Simulation - Helix; the
prompt latency corresponds to Figure 5(g)'s Simulation - Helix; the decode latency corresponds
to Figure 5(h)'s Simulation - Helix.
```bash
python step3_simulation.py helix llama70b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B online simulation results: Helix
Total decode throughput: 141.6 tokens/s
Prompt latency:
Latency 5th percentile: 0.80 s
Latency 25th percentile: 1.38 s
Latency 50th percentile: 2.64 s
Latency 75th percentile: 3.52 s
Latency 95th percentile: 4.41 s
Decode latency:
Latency 5th percentile: 0.56 s
Latency 25th percentile: 0.76 s
Latency 50th percentile: 0.98 s
Latency 75th percentile: 1.67 s
Latency 95th percentile: 3.19 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama70b/ilp_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(11) Run LLaMA 70B in online setup using Swarm and observe its decode throughput, prompt latency
and decode latency. The decode throughput corresponds to Figure 5(d)'s Simulation - Swarm; the
prompt latency corresponds to Figure 5(g)'s Simulation - Swarm; the decode latency corresponds
to Figure 5(h)'s Simulation - Swarm.
```bash
python step3_simulation.py swarm llama70b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B online simulation results: Swarm
Total decode throughput: 69.8 tokens/s
Prompt latency:
Latency 5th percentile: 0.95 s
Latency 25th percentile: 1.70 s
Latency 50th percentile: 3.32 s
Latency 75th percentile: 4.04 s
Latency 95th percentile: 5.01 s
Decode latency:
Latency 5th percentile: 0.57 s
Latency 25th percentile: 0.82 s
Latency 50th percentile: 1.04 s
Latency 75th percentile: 1.59 s
Latency 95th percentile: 2.51 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama70b/swarm_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(12) Run LLaMA 70B in online setup using Separate Pipelines and observe its decode throughput,
prompt latency and decode latency. The decode throughput corresponds to Figure 5(d)'s Simulation -
Separate Pipelines (SP); the prompt latency corresponds to Figure 5(g)'s Simulation - Separate
Pipelines (SP); the decode latency corresponds to Figure 5(h)'s Simulation - Separate Pipelines (SP).
```bash
python step3_simulation.py separate llama70b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B online simulation results: Separate
Total decode throughput: 84.0 tokens/s
Prompt latency:
Latency 5th percentile: 0.19 s
Latency 25th percentile: 0.36 s
Latency 50th percentile: 0.80 s
Latency 75th percentile: 1.48 s
Latency 95th percentile: 4.19 s
Decode latency:
Latency 5th percentile: 0.14 s
Latency 25th percentile: 0.14 s
Latency 50th percentile: 0.22 s
Latency 75th percentile: 0.69 s
Latency 95th percentile: 1.20 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama70b/separate_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

### Step 4: Generate Real System Config Files

TODO!!

### Step 5: Run Prototype System Experiments

TODO!!

## Section 6.4 Geo-Distributed Clusters

All files related to this group of experiments are located in `artifact_evaluation/distributed_clusters`.
Start from the root directory of the repository:
```bash
cd artifact_evaluation/distributed_clusters
```
The workflow is similar to the single cluster experiments.

### Step 1: Generate Cluster Config Files
Run the following command to generate cluster config files:
```bash
python step1_gen_cluster.py
```
This will automatically generate the config file that represents a geo-distributed cluster with
24 machines (4 1xA100 machines, 8 1xL4 machines, 12 1xT4 machines) in `config/3cluster24.ini`.
The cluster contains three regions. In region 1, there are 4 A100 machines; in region 2, there
are 2 L4 machines and 8 T4 machines; in region 3, there are 6 L4 machines and 4 T4 machines.
It also creates the config files that represent a sub-cluster formed by each type of machine:
`config/a100.ini`, `config/l4.ini` and `config/t4.ini`.

### Step 2: Model Placement

The next step is to generate the model placement for the cluster. Run the following commands
to generate model placements for LLaMA-1 30B and LLaMA-2 70B using different model placement
methods. Notice that before running Helix's MILP-based model placement planner, you need to
remove the `./layout_llama30b/ilp` and `./layout_llama70b/ilp` directories, which currently
contains the result we get. We suggest moving them to a backup place if you want to compare
your results with ours. Also, for `llama70b`, you need to run `swarm` before running
Helix's MILP model placement planner, as we bootstrap the solver with `swarm`'s solution.

> **Notes:** We notice that Gurobi produces completely different optimization traces when
> using **different licenses, even when using the same random seed**. When using the default
> limited license, the optimization performance is much worse than that of using the academic
> license. (objective value = 952 v.s. 1212) Unfortunately, we are not allowed and unable to
> bind our academic Gurobi license to the cluster provided. Note that this is an issue with
> Gurobi **licensing** instead of our system, and we provide our optimization trace in
> `./layout_llama70b/ilp/trace.txt` for you to compare against."

> **Notes:** Running Helix to search for a model placement for LLaMA 70B may take a long time.
> We set the max running time to 10 hours, but you can stop the solver at any time with `ctrl +c`. 
> In our experiments, on a machine with 14 cores and academic license, we manually early-stop
> the solver at round 45 minutes. This solution at this point already has good quality. The
> objective value (Incumbent) equals to 1212.

```bash
# Generate model placement using heuristic method Petals
python step2_gen_layout.py petals llama30b
python step2_gen_layout.py petals llama70b
# Generate model placement using heuristic method Swarm
python step2_gen_layout.py swarm llama30b
python step2_gen_layout.py swarm llama70b
# Generate model placement using heuristic method separate pipelines
python step2_gen_layout.py separate llama30b
python step2_gen_layout.py separate llama70b
# Generate model placement using Helix's MILP-based method
python step2_gen_layout.py ilp llama30b  # remove ./layout_llama30b/ilp before running
python step2_gen_layout.py ilp llama70b  # remove ./layout_llama70b/ilp before running
```

After running the commands above, you will get model placement files located in:
+ **petals & llama 30b**: `./layout_llama30b/petals` (`petals_sol.ini` and `simulator_cluster.ini`)
+ **petals & llama 70b**: `./layout_llama70b/petals` (`petals_sol.ini` and `simulator_cluster.ini`)
+ **swarm & llama 30b**: `./layout_llama30b/swarm` (`swarm_sol.ini` and `simulator_cluster.ini`)
+ **swarm & llama 70b**: `./layout_llama70b/swarm` (`swarm_sol.ini` and `simulator_cluster.ini`)
+ **separate & llama 30b**: `./layout_llama30b/separate` (6 manually created files)
+ **separate & llama 70b**: `./layout_llama70b/separate` (6 manually created files)
+ **ilp & llama 30b**: `./layout_llama30b/ilp/a100`, `./layout_llama30b/ilp/l4`, `./layout_llama30b/ilp/t4`, each containing the model placement for a sub-cluster
+ **ilp & llama 70b**: `./layout_llama70b/ilp` (`ilp_sol.ini` and `simulator_cluster.ini`, and 3 other files that records information about the MILP problem)

### Step 3: Run Simulation
With the model placement files generated, we can run the simulation and reproduce the results in the
paper.

(1) Run LLaMA 30B in offline setup using Helix and observe its decode throughput. This
corresponds to Figure 7(a)'s offline - Helix in the paper.
```bash
python step3_simulation.py helix llama30b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B offline simulation results: Helix
Total decode throughput: 232.7 tokens/s
************************************************************
```
The result here is slightly different from the one in the paper, as the model placement
found is slightly different.

(2) Run LLaMA 30B in offline setup using Swarm and observe its decode throughput. This
corresponds to Figure 7(a)'s offline - Swarm in the paper.
```bash
python step3_simulation.py swarm llama30b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B offline simulation results: Swarm
Total decode throughput: 98.7 tokens/s
************************************************************
```

(3) Run LLaMA 30B in offline setup using Separate Pipelines and observe its decode
throughput. This corresponds to Figure 7(a)'s offline - Separate Pipelines (SP) in the paper.
```bash
python step3_simulation.py separate llama30b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B offline simulation results: Separate
Total decode throughput: 223.2 tokens/s
************************************************************
```

(4) Run LLaMA 30B in online setup using Helix and observe its decode throughput, prompt latency
and decode latency. The decode throughput corresponds to Figure 7(a)'s online - Helix; the
prompt latency corresponds to Figure 7(c)'s Helix (H); the decode latency corresponds to Figure
7(d)'s Helix (H).
```bash
python step3_simulation.py helix llama30b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B online simulation results: Helix
Total decode throughput: 183.5 tokens/s
Prompt latency:
Latency 5th percentile: 0.18 s
Latency 25th percentile: 0.25 s
Latency 50th percentile: 0.51 s
Latency 75th percentile: 1.37 s
Latency 95th percentile: 4.41 s
Decode latency:
Latency 5th percentile: 0.16 s
Latency 25th percentile: 0.18 s
Latency 50th percentile: 0.23 s
Latency 75th percentile: 0.47 s
Latency 95th percentile: 0.83 s
************************************************************
```
The result here is slightly different from the one in the paper, as the model placement
found is slightly different.

We also store the raw latency distribution files as pickle files in
`./simulation_llama30b/ilp_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(5) Run LLaMA 30B in online setup using Swarm and observe its decode throughput, prompt latency
and decode latency. The decode throughput corresponds to Figure 7(a)'s online - Swarm; the
prompt latency corresponds to Figure 7(c)'s Swarm (S); the decode latency corresponds to Figure
7(d)'s Swarm (S).
```bash
python step3_simulation.py swarm llama30b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B online simulation results: Swarm
Total decode throughput: 82.1 tokens/s
Prompt latency:
Latency 5th percentile: 0.98 s
Latency 25th percentile: 1.82 s
Latency 50th percentile: 3.25 s
Latency 75th percentile: 5.10 s
Latency 95th percentile: 7.32 s
Decode latency:
Latency 5th percentile: 0.38 s
Latency 25th percentile: 0.43 s
Latency 50th percentile: 0.48 s
Latency 75th percentile: 0.53 s
Latency 95th percentile: 1.12 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama30b/swarm_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(6) Run LLaMA 30B in online setup using Separate Pipelines and observe its decode throughput,
prompt latency and decode latency. The decode throughput corresponds to Figure 7(a)'s online -
Separate Pipelines (SP); the prompt latency corresponds to Figure 7(c)'s Separate Pipelines (SP);
the decode latency corresponds to Figure 7(d)'s Separate Pipelines (SP).
```bash
python step3_simulation.py separate llama30b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa30B online simulation results: Separate
Total decode throughput: 173.8 tokens/s
Prompt latency:
Latency 5th percentile: 0.18 s
Latency 25th percentile: 0.25 s
Latency 50th percentile: 0.50 s
Latency 75th percentile: 1.72 s
Latency 95th percentile: 6.47 s
Decode latency:
Latency 5th percentile: 0.16 s
Latency 25th percentile: 0.18 s
Latency 50th percentile: 0.23 s
Latency 75th percentile: 0.46 s
Latency 95th percentile: 0.92 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama30b/separate_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(7) Run LLaMA 70B in offline setup using Helix and observe its decode throughput. This
corresponds to Figure 7(b)'s offline - Helix in the paper.
```bash
python step3_simulation.py helix llama70b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B offline simulation results: Helix
Total decode throughput: 98.0 tokens/s
************************************************************
```
    
(8) Run LLaMA 70B in offline setup using Swarm and observe its decode throughput. This
corresponds to Figure 7(b)'s offline - Swarm in the paper.
```bash
python step3_simulation.py swarm llama70b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B offline simulation results: Swarm
Total decode throughput: 51.9 tokens/s
************************************************************
```

(9) Run LLaMA 70B in offline setup using Separate Pipelines and observe its decode throughput.
This corresponds to Figure 7(b)'s offline - Separate Pipelines (SP) in the paper.
```bash
python step3_simulation.py separate llama70b offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B offline simulation results: Separate
Total decode throughput: 61.9 tokens/s
************************************************************
```

(10) Run LLaMA 70B in online setup using Helix and observe its decode throughput, prompt latency
and decode latency. The decode throughput corresponds to Figure 7(b)'s online - Helix; the
prompt latency corresponds to Figure 7(e)'s Helix (H); the decode latency corresponds to Figure
7(f)'s Helix (H).
```bash
python step3_simulation.py helix llama70b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B online simulation results: Helix
Total decode throughput: 59.3 tokens/s
Prompt latency:
Latency 5th percentile: 1.50 s
Latency 25th percentile: 2.63 s
Latency 50th percentile: 5.79 s
Latency 75th percentile: 7.53 s
Latency 95th percentile: 9.41 s
Decode latency:
Latency 5th percentile: 0.60 s
Latency 25th percentile: 0.79 s
Latency 50th percentile: 0.93 s
Latency 75th percentile: 2.34 s
Latency 95th percentile: 7.72 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama70b/ilp_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(11) Run LLaMA 70B in online setup using Swarm and observe its decode throughput, prompt latency
and decode latency. The decode throughput corresponds to Figure 7(b)'s online - Swarm; the
prompt latency corresponds to Figure 7(e)'s Swarm (S); the decode latency corresponds to Figure
7(f)'s Swarm (S).
```bash
python step3_simulation.py swarm llama70b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B online simulation results: Swarm
Total decode throughput: 31.0 tokens/s
Prompt latency:
Latency 5th percentile: 2.11 s
Latency 25th percentile: 3.96 s
Latency 50th percentile: 7.22 s
Latency 75th percentile: 10.71 s
Latency 95th percentile: 14.83 s
Decode latency:
Latency 5th percentile: 0.77 s
Latency 25th percentile: 0.97 s
Latency 50th percentile: 1.23 s
Latency 75th percentile: 2.39 s
Latency 95th percentile: 6.50 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama70b/swarm_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

(12) Run LLaMA 70B in online setup using Separate Pipelines and observe its decode throughput,
prompt latency and decode latency. The decode throughput corresponds to Figure 7(b)'s online -
Separate Pipelines (SP); the prompt latency corresponds to Figure 7(e)'s Separate Pipelines (SP);
the decode latency corresponds to Figure 7(f)'s Separate Pipelines (SP).
```bash
python step3_simulation.py separate llama70b online
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B online simulation results: Separate
Total decode throughput: 33.6 tokens/s
Prompt latency:
Latency 5th percentile: 1.27 s
Latency 25th percentile: 2.20 s
Latency 50th percentile: 4.96 s
Latency 75th percentile: 6.02 s
Latency 95th percentile: 15.82 s
Decode latency:
Latency 5th percentile: 0.71 s
Latency 25th percentile: 0.97 s
Latency 50th percentile: 1.20 s
Latency 75th percentile: 1.37 s
Latency 95th percentile: 5.01 s
************************************************************
```
We also store the raw latency distribution files as pickle files in
`./simulation_llama70b/separate_online`. You can refer to `analyze_latency` in `step3_simulation.py`
if you want to parse and check them.

## Section 6.5 High GPU-Heterogeneity Cluster
All files related to this group of experiments are located in `artifact_evaluation/high_heterogeneity`.
Start from the root directory of the repository:
```bash
cd artifact_evaluation/high_heterogeneity
```
The workflow is similar to the first two groups of experiments.

### Step 1: Generate Cluster Config Files
Run the following command to generate cluster config files:
```bash
python step1_gen_cluster.py
```
This will automatically generate the config file that represents a single cluster with
42 machines (4 1xA100 machines, 6 1xV100 machines, 8 1xL4 machines, 4 2xL4 machines, 10 1xT4 machines,
6 2xT4 machines and 4 4xT4 machines) in `config/cluster42.ini`.

### Step 2: Model Placement

The next step is to generate the model placement for the cluster. Different from the previous
two groups of experiments, we only deploy LLaMA-2 70B in this 42-node cluster. Run the following
commands to generate model placements using different model placement methods. Notice that before
running Helix's MILP-based model placement planner, you need to remove the `./layout_llama70b/ilp`
directories, which currently contains the result we get. We suggest moving them to a backup place
if you want to compare your results with ours. Also, for `llama70b`, you need to run `swarm`
before running Helix's MILP model placement planner, as we bootstrap the solver with `swarm`'s
solution.

> **Notes:** The default limited license for Gurobi has a size limit on the MILP problem. The size
> of the MILP problem for LLaMA 70B on this 42-node cluster exceeds the limit. Gurobi will refuse
> to solve and throw an error: "gurobipy.GurobiError: Model too large for size-limited license;
> visit https://gurobi.com/unrestricted for more information". Note that this is an issue with
> Gurobi **licensing** instead of our system, and we provide our optimization trace in
> `./layout_llama70b/ilp/trace.txt` for you to compare against.

> **Notes:** Running Helix to search for a model placement for LLaMA 70B may take a long time.
> We set the max running time to 10 hours, but you can stop the solver at any time with `ctrl +c`. 
> In our experiments, on a machine with 14 cores and academic license, we manually early-stop
> the solver at round 50 minutes. This solution at this point already has good quality. The
> objective value (Incumbent) equals to 4137.

```bash
# Generate model placement using heuristic method Petals
python step2_gen_layout.py petals
# Generate model placement using heuristic method Swarm
python step2_gen_layout.py swarm
# Generate model placement using heuristic method separate pipelines
python step2_gen_layout.py separate
# Generate model placement using Helix's MILP-based method
python step2_gen_layout.py ilp  # remove ./layout_llama70b/ilp before running
```

After running the commands above, you will get model placement files located in:
+ **petals**: `./layout_llama70b/petals` (`petals_sol.ini` and `simulator_cluster.ini`)
+ **swarm**: `./layout_llama70b/swarm` (`swarm_sol.ini` and `simulator_cluster.ini`)
+ **separate**: `./layout_llama70b/separate` (12 manually created files in 6 directories)
+ **ilp**: `./layout_llama70b/ilp` (`ilp_sol.ini` and `simulator_cluster.ini`, and 4 other files that records information about the MILP problem)

### Step 3: Run Simulation
With the model placement files generated, we can run the simulation and reproduce the results in the
paper.

(1) Run LLaMA 70B in offline setup using Helix and observe its decode throughput. This
corresponds to Figure 8(a)'s offline - Helix in the paper.
```bash
python step3_simulation.py helix offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B offline simulation results: Helix
Total decode throughput: 645.6 tokens/s
************************************************************
```
The result here is slightly higher than the result in the paper, because of the slight
difference in the model placement.
    
(2) Run LLaMA 70B in offline setup using Swarm and observe its decode throughput. This
corresponds to Figure 8(a)'s offline - Swarm in the paper.
```bash
python step3_simulation.py swarm offline
```
After running the simulation, you will see a log like the following at the end:
```
************************************************************
LLaMa70B offline simulation results: Swarm
Total decode throughput: 473.2 tokens/s
************************************************************
```

(3) Run LLaMA 70B in offline setup using Separate Pipelines and observe its decode throughput.
This corresponds to Figure 8(a)'s offline - Separate Pipelines (SP) in the paper.
```bash
python step3_simulation.py separate offline
```
After running the simulation, you will see a log like the following at the end:
```

```

(4) Run LLaMA 70B in offline setup using Separate Pipelines Plus and observe its decode throughput.
This corresponds to Figure 8(a)'s offline - Separate Pipelines Plus (SP+) in the paper.
```bash
python step3_simulation.py sp_plus offline
```
After running the simulation, you will see a log like the following at the end:
```

```
