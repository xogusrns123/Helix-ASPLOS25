#!/bin/bash
# docker_run.sh

sudo docker run --gpus all -it --network host \
-v $(pwd)/llm_sys:/home/kth/helix/llm_sys \
-v $(pwd)/examples:/home/kth/helix/examples \
-v $(pwd)/simulator:/home/kth/helix/simulator \
-v $(pwd)/artifact_evaluation:/home/kth/helix/artifact_evaluation \
-v $(pwd)/setup.py:/home/kth/helix/setup.py \
-v $(pwd)/readme.md:/home/kth/helix/readme.md \
helix:latest