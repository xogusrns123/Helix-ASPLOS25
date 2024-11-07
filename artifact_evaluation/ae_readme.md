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
the your results with ours. Also, for `llama70b`, you need to run `petals` before running
Helix's MILP model placement planner, as we bootstrap the solver with `petals`' solution.

> **Notes:** Running Helix to search for a model placement for LLaMA 70B may take a long time.
> We set the max running time to 10 hours, but you can stop the solver at any time with `ctrl +c`. 
> In our experiments, on a machine with 14 cores and academic license, we manually early-stop
> the solver at round 10 minutes. This solution at this point already has good quality. The
> objective value (Incumbent) equals to 1289.

> **Notes:** We notice that Gurobi produces completely different optimization traces when
> using **different licenses, even when using the same random seed**. When using the default
> limited license, the optimization performance is much worse than that of using the academic
> license. (objective value = 952 v.s. 1289) Unfortunately, we are not allowed and unable to
> bind our academic Gurobi license to the cluster provided. We want to state that this is an
> issue with Gurobi instead of our system, and we provide our optimization trace in
> `./layout_llama70b/ilp/trace.txt` for you to compare against.

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

TODO: describe the steps and result & files

