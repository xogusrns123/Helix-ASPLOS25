# 2024.11.02 Yixuan Mei

# 2024.01.09 Lee JiHyuk
# Adding profiling to worker

import sys

from llm_sys.worker import run_worker

import os

def main():
    # parse arguments
    if len(sys.argv) != 2:
        print("Usage: python step3_start_worker.py <scheduling_method>")
        print("  scheduling_method: maxflow | swarm | random")
        return
    scheduling_method = sys.argv[1]

    # check arguments
    assert scheduling_method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {scheduling_method}!"
    print(f"Starting worker with scheduling method: {scheduling_method}.")

    # Added by LJH
    result_dir = f"./result/worker/"
    os.makedirs(result_dir, exist_ok=True)
    
    # run worker
    run_worker(scheduling_method=scheduling_method, model_name="./Llama-2-7b-hf", result_logging_dir=result_dir)


if __name__ == '__main__':
    main()
