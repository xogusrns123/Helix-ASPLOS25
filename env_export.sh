#!/bin/bash

# Start and end of the range
START=70000
END=70020
VALUE_START=6000

# Loop through the range and set environment variables
for ((i=START, value=VALUE_START; i<=END; i++, value++)); do
    echo "export VAST_TCP_PORT_$i=$value" >> ~/env_vars.sh
    echo "Exported: VAST_TCP_PORT_$i=$value"
done