from llm_sys.gen_sys_config import gen_sys_config
from simulator.event_simulator.cluster_simulator import ModelName


def main():
    gen_sys_config(
        host_ip="10.148.15.200",
        type2ips={"A100": ["10.148.0.25", "10.148.0.26", "10.148.0.28", "10.148.0.33"],
                  "L4": ["10.148.0.34", "10.148.0.35", "10.148.0.36", "10.148.0.37",
                         "10.148.0.38", "10.148.0.39", "10.148.0.45", "10.148.0.47"],
                  "T4": ["10.148.0.48", "10.148.0.49", "10.148.0.50", "10.148.0.51",
                         "10.148.0.52", "10.148.0.53", "10.148.0.58", "10.148.0.59",
                         "10.148.0.62", "10.148.15.194", "10.148.15.193", "10.148.15.202"]},
        # model and machine
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12},
        model_name=ModelName.LLaMa70B,
        # cluster
        complete_cluster_file_name="./config/single24.ini",
        machine_profile_file_name="./config/machine_profile.ini",
        # model placement
        solution_file_name="./layout/ilp_sol.ini",
        simulator_cluster_file_name="./layout/simulator_cluster.ini",
        # output directory
        output_dir="./config",
        output_file_name="real_sys_config.txt"
    )
    print("System config generated to ./config/real_sys_config.txt!")


if __name__ == '__main__':
    main()
