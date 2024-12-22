#!/bin/bash

# Path to the node list file
NODES_FILE="./examples/real_sys/config/real_sys_config.txt"

# Name of the Docker container to stop
CONTAINER_NAME="helix-container"

# Sudo password (replace 'your_password' with the actual password)
SUDO_PASSWORD="th28620720!"

# Check if the node list file exists
if [ ! -f "$NODES_FILE" ]; then
    echo "Node list file ($NODES_FILE) does not exist."
    exit 1
fi

# Parse IP addresses and stop the Docker container on each node sequentially
grep "ip_address:" "$NODES_FILE" | awk '{print $2}' | while read -r ip; do
    echo "[$ip]: Stopping container '$CONTAINER_NAME'..."
    ssh "$ip" "echo $SUDO_PASSWORD | sudo -S docker stop $CONTAINER_NAME"
    if [ $? -eq 0 ]; then
        echo "[$ip]: Successfully stopped container '$CONTAINER_NAME'."
    else
        echo "[$ip]: Failed to stop container '$CONTAINER_NAME'."
    fi
done

echo "Docker stop completed on all nodes sequentially."

# Parse IP addresses and remove the Docker container on each node sequentially
grep "ip_address:" "$NODES_FILE" | awk '{print $2}' | while read -r ip; do
    echo "[$ip]: Stopping container '$CONTAINER_NAME'..."
    ssh "$ip" "echo $SUDO_PASSWORD | sudo -S docker rm $CONTAINER_NAME"
    if [ $? -eq 0 ]; then
        echo "[$ip]: Successfully stopped container '$CONTAINER_NAME'."
    else
        echo "[$ip]: Failed to stop container '$CONTAINER_NAME'."
    fi
done

echo "Docker remove completed on all nodes sequentially."