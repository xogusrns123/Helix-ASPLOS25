# 2024.10.29 Yixuan Mei
import sys
import os
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer, ModelName
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def petals_layout(model_name: ModelName, save_path: str):
    # heuristic method: petals
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster24.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=model_name,
        workspace_path=save_path,
        layout_method=LayoutMethod.Petals,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )
    petals_args = {
        "seed": 0,
        "max_out_links_per_node": 24,
    }
    layout_synthesizer.synthesize(args=petals_args)


def swarm_layout(model_name: ModelName, save_path: str, num_stages: int):
    # heuristic method: swarm
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/cluster24.ini",
        machine_profile_name="./config/machine_profiles.ini",
        model_name=model_name,
        workspace_path=save_path,
        layout_method=LayoutMethod.Swarm,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )
    swarm_args = {
        "seed": 0,
        "num_stages": num_stages,  # as few as possible
        "max_out_links_per_node": 24,
    }
    layout_synthesizer.synthesize(args=swarm_args)


def main():
    assert len(sys.argv) == 3, f"Usage: python {sys.argv[0]} <ilp/swarm/petals/homogeneous> <llama30b/llama70b>"
    layout_method = sys.argv[1]
    model_name = sys.argv[2]

    if layout_method == "ilp":
        # TODO
        raise NotImplementedError
    elif layout_method == "swarm":
        # Heuristic method: swarm
        if model_name == "llama30b":
            os.makedirs("./layout_llama30b/swarm", exist_ok=True)
            swarm_layout(ModelName.LLaMa30B, "./layout_llama30b/swarm", 10)
            print("Layout for LLaMa30B model is generated using Swarm method.")

        elif model_name == "llama70b":
            os.makedirs("./layout_llama70b/swarm", exist_ok=True)
            swarm_layout(ModelName.LLaMa70B, "./layout_llama70b/swarm", 20)
            print("Layout for LLaMa70B model is generated using Swarm method.")

        else:
            raise ValueError(f"Invalid model name: {model_name}")

    elif layout_method == "petals":
        # Heuristic method: petals
        if model_name == "llama30b":
            os.makedirs("./layout_llama30b/petals", exist_ok=True)
            petals_layout(ModelName.LLaMa30B, "./layout_llama30b/petals")
            print("Layout for LLaMa30B model is generated using Petals method.")

        elif model_name == "llama70b":
            os.makedirs("./layout_llama70b/petals", exist_ok=True)
            petals_layout(ModelName.LLaMa70B, "./layout_llama70b/petals")
            print("Layout for LLaMa70B model is generated using Petals method.")

        else:
            raise ValueError(f"Invalid model name: {model_name}")

    elif layout_method == "homogeneous":
        # TODO
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid layout method: {layout_method}")


if __name__ == '__main__':
    main()
