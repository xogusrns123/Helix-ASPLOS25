#!/bin/bash

# List of machines where you want to deploy Docker containers.
# Replace these with your actual machine names or IPs.
MACHINES=("143.248.53.59" "143.248.53.25" "143.248.53.45")

# Remote path where you want to place docker_run.sh on each machine.
REMOTE_PATH="/home/kth/brl/Helix-ASPLOS25/docker_run.sh"

# Iterate over each machine in the list.
for HOST in "${MACHINES[@]}"; do
    echo "================================================================="
    echo "Deploying Docker containers on ${HOST}..."
    echo "================================================================="
    
    # 1. SSH into the remote machine and run the script under sudo.
    #    The `-t` flag ensures we allocate a pseudo-TTY, so you can enter the sudo password.
    ssh -t "${HOST}" "chmod +x ${REMOTE_PATH} && sudo ${REMOTE_PATH}"

    echo
done

echo "All deployments finished!"
