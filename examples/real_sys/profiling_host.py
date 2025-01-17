# 2024.11.02 Yixuan Mei
import os
import sys

from llm_sys.heuristic_host import run_heuristic_host_profiling
from simulator.event_simulator.cluster_simulator import ModelName

def heuristic_profiling(duration:int, batch_size:int, seq_len:int, output_len:int, num_node: str):
    # run heuristic host online
    print(f"Running profiling with")
    
    run_heuristic_host_profiling(
        duration=duration,
        initial_launch_num=batch_size,
        seq_len=seq_len,
        output_len=output_len,
        device_num=int(num_node)
    )

def main():
    # parse arguments
    if len(sys.argv) != 4:
        print("Usage: python3 profiling_host.py <batch_size> <num_nodes> <duration>")
        print("  batch_size: Number of requests to be processed at once")
        print("  num_nodes: Total number of nodes to use in the cluster")
        print("  duration: Time to run profiling")
        return
    
    batch_size = int(sys.argv[1])
    num_nodes = int(sys.argv[2])
    duration = int(sys.argv[3])
    
    heuristic_profiling(batch_size=batch_size, seq_len=0, output_len=0, num_node=num_nodes, duration=duration)


if __name__ == '__main__':
    main()
