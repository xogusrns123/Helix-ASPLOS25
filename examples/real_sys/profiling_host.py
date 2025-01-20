# 2025.1.18 Lee JiHyuk

import sys
import argparse

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

# Model selction from host should added!!
def main():
    # Parse arguments using argparse
    parser = argparse.ArgumentParser(description="Run heuristic profiling.")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of requests to process at once")
    parser.add_argument("--num_nodes", type=int, required=True, help="Total number of nodes to use in the cluster")
    parser.add_argument("--duration", type=int, required=True, help="Time to run profiling (seconds)")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length")
    parser.add_argument("--output_len", type=int, required=True, help="Output length")
    
    args = parser.parse_args()
    
    heuristic_profiling(
        batch_size=args.batch_size, 
        seq_len=args.seq_len, 
        output_len=args.output_len, 
        num_node=args.num_nodes, 
        duration=args.duration
    )

# python profiling_host.py --batch_size 32 --num_nodes 3 --duration 90 --seq_len 1000 --output_len 125

if __name__ == '__main__':
    main()
