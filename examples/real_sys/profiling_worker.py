# 2024.11.02 Yixuan Mei

# 2024.01.09 Lee JiHyuk
# Adding profiling to worker

import argparse

from llm_sys.worker import run_worker

import os

def main():
    parser = argparse.ArgumentParser(description="Run heuristic profiling.")
    parser.add_argument("--num_nodes", type=int, required=True, help="Total number of nodes to use in the cluster")

    args = parser.parse_args()
    
    scheduling_method = "random"
    print(f"Starting worker with scheduling method: {scheduling_method}.")

    # Added by LJH
    result_dir = f"./profiling/{scheduling_method}_worker/"
    os.makedirs(result_dir, exist_ok=True)
    
    worker_config_file_path = f"./config/device_config.txt"
    
    # run worker
    run_worker(
        scheduling_method=scheduling_method, 
        model_name="./Llama-2-7b-hf", 
        result_logging_dir=result_dir, 
        worker_config_file_path=worker_config_file_path, 
        device_num=args.num_nodes
    )


if __name__ == '__main__':
    main()
