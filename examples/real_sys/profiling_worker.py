# 2024.11.02 Yixuan Mei

# 2024.01.09 Lee JiHyuk
# Adding profiling to worker

import argparse

from llm_sys.worker import run_worker

import os

def main():
    parser = argparse.ArgumentParser(description="Run heuristic profiling.")
    parser.add_argument("--num_nodes", type=int, required=True, help="Total number of nodes to use in the cluster")
    parser.add_argument("--model", type=str, required=True, help="Total number of nodes to use in the cluster")

    args = parser.parse_args()
    
    if args.model == "7b":
        model = "./Llama-2-7b-hf"
    elif args.model == "13b":
        model = "./Llama-2-13b-hf"
    elif args.model == "70b":
        model = "./Llama-2-70b-hf"
        
    result_dir = f"./profiling/random_worker/"
    os.makedirs(result_dir, exist_ok=True)
    
    worker_config_file_path = f"./config/device_config.txt"
    
    # run worker
    run_worker(
        scheduling_method="random", 
        model_name=model, 
        result_logging_dir=result_dir, 
        worker_config_file_path=worker_config_file_path, 
        device_num=args.num_nodes
    )

# python profiling_worker.py --num_nodes 3 --model 7b

if __name__ == '__main__':
    main()
