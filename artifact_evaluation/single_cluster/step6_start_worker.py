# 2024.11.02 Yixuan Mei
import sys

from llm_sys.worker import run_worker


def main():
    # parse arguments
    if len(sys.argv) != 3:
        print("Usage: python step3_start_worker.py <llama30b/llama70b> <scheduling_method>")
        print("  scheduling_method: maxflow | swarm | random")
        return
    model_name = sys.argv[1]
    scheduling_method = sys.argv[2]

    # check arguments
    assert scheduling_method in ["maxflow", "swarm", "random"], f"Unsupported scheduling method: {scheduling_method}!"
    assert model_name in ["llama30b", "llama70b"], f"Invalid model name: {model_name}"
    print(f"Starting worker with scheduling method: {scheduling_method}.")
    print(f"Model: {model_name}")

    # run worker
    model_path = f"./models/{model_name}"
    run_worker(scheduling_method=scheduling_method, model_name=model_path)


if __name__ == '__main__':
    main()