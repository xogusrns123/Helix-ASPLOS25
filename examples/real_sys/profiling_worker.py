# 2024.11.02 Yixuan Mei

# 2024.01.09 Lee JiHyuk
# Adding profiling to worker

import sys

from llm_sys.worker import run_worker

import os

def main():
    # parse arguments
    if len(sys.argv) != 3:
        print("Usage: python step3_start_worker.py <scheduling_method>")
        print("  num_nodes: int")
        return
    num_nodes = sys.argv[1]

    scheduling_method = "random"
    print(f"Starting worker with scheduling method: {scheduling_method}.")

    # Added by LJH
    result_dir = f"./profiling/{scheduling_method}_worker/"
    os.makedirs(result_dir, exist_ok=True)
    
    worker_config_file_path = f"./config/device_config.txt"
    
    # run worker
    run_worker(scheduling_method=scheduling_method, 
               model_name="./Llama-2-7b-hf", 
               result_logging_dir=result_dir, 
               worker_config_file_path=worker_config_file_path, 
               device_num=int(num_nodes))


if __name__ == '__main__':
    main()
