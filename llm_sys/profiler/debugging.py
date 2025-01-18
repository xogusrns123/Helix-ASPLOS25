import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple


def calculate_prefill_costs(prefill_df, node_path, worker_nodes):
    """
    Calculate prefill communication and computation costs.

    :param prefill_df: DataFrame containing prefill data
    :param node_path: List of (src, dst) tuples representing communication paths
    :param worker_nodes: List of worker node indices
    :return: Tuple of (prefill_comm_costs, prefill_comp_costs)
    """
    # Communication cost: (src, dst) -> float
    prefill_comm_costs: Dict[Tuple[int, int], float] = defaultdict(float)
    # Computation cost: node_index -> float
    prefill_comp_costs: Dict[int, float] = defaultdict(float)

    for src, dst in node_path:
        src_cond = (prefill_df['in_out'] == 'out') & (prefill_df['source'] == src)
        dst_cond = (prefill_df['in_out'] == 'in') & (prefill_df['source'] == dst)

        src_row = prefill_df[src_cond]
        dst_row = prefill_df[dst_cond]

        if not src_row.empty and not dst_row.empty:
            prefill_comm_costs[(src, dst)] = dst_row.iloc[0]['time_stamp'] - src_row.iloc[0]['time_stamp']

    for node in worker_nodes:
        in_cond = (prefill_df['in_out'] == 'in') & (prefill_df['source'] == node)
        out_cond = (prefill_df['in_out'] == 'out') & (prefill_df['source'] == node)

        in_row = prefill_df[in_cond]
        out_row = prefill_df[out_cond]

        if not in_row.empty and not out_row.empty:
            prefill_comp_costs[node] = out_row.iloc[0]['time_stamp'] - in_row.iloc[0]['time_stamp']

    return prefill_comm_costs, prefill_comp_costs

def calculate_decode_costs(decode_df, node_path, worker_nodes):
    """
    Calculate decode communication and computation costs.

    :param decode_df: DataFrame containing decode data
    :param node_path: List of (src, dst) tuples representing communication paths
    :param worker_nodes: List of worker node indices
    :return: Tuple of (decode_comm_costs, decode_comp_costs)
    """
    # Communication cost: (src, dst) -> List[float]
    decode_comm_costs: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    # Computation cost: node_index -> List[float]
    decode_comp_costs: Dict[int, List[float]] = defaultdict(list)

    for src, dst in node_path:
        src_cond = (decode_df['in_out'] == 'out') & (decode_df['source'] == src)
        dst_cond = (decode_df['in_out'] == 'in') & (decode_df['source'] == dst)

        src_rows = decode_df[src_cond].reset_index(drop=True)
        dst_rows = decode_df[dst_cond].reset_index(drop=True)

        min_len = min(len(src_rows), len(dst_rows))
        for i in range(min_len):
            src_time = src_rows.loc[i, 'time_stamp']
            dst_time = dst_rows.loc[i, 'time_stamp']
            decode_comm_costs[(src, dst)].append(dst_time - src_time)

    for node in worker_nodes:
        in_cond = (decode_df['in_out'] == 'in') & (decode_df['source'] == node)
        out_cond = (decode_df['in_out'] == 'out') & (decode_df['source'] == node)

        in_rows = decode_df[in_cond].reset_index(drop=True)
        out_rows = decode_df[out_cond].reset_index(drop=True)

        # Ensure in_rows and out_rows have the same length for 1:1 mapping
        min_len = min(len(in_rows), len(out_rows))
        for i in range(min_len):
            in_time = in_rows.loc[i, 'time_stamp']
            out_time = out_rows.loc[i, 'time_stamp']
            decode_comp_costs[node].append(out_time - in_time)

    return decode_comm_costs, decode_comp_costs


def main():
    merged_df = pd.read_csv("./debugging.csv")

    node_path = [(0, 1), (1, 2), (2, 0)]
    worker_nodes = [1, 2]

    # 2. Iterate through every request_ids
    request_id_list = merged_df['request_id'].unique().tolist()

    for request_id in request_id_list:
        # 2-1. Filter merged_df by request_id
        request_condition = merged_df["request_id"] == request_id
        request_df = merged_df[request_condition].reset_index(drop=True)

        if request_df.empty:
            print(f"Request ID {request_id} has no filtered DataFrame")
            continue

        # 2-2. Find criterion line
        decode_start_condition = (request_df['in_out'] == 'out') & \
                                 (request_df['mode'] == 'decode') & \
                                 (request_df['source'] == 0)
        decode_start_index = request_df.index[decode_start_condition].min()
        print(f"request_id {request_id}:", decode_start_index)

        # 2-3. Divide DataFrame by prefill and decode
        if pd.notna(decode_start_index):
            prefill_df = request_df.iloc[:decode_start_index].reset_index(drop=True)
            decode_df = request_df.iloc[decode_start_index:].reset_index(drop=True)
        else:
            print(f"No rows satisfy the start condition for request_id {request_id}.")
            continue

        # print(prefill_df)

        # 2-4. Calculate prefill costs
        # prefill_comm_costs, prefill_comp_costs = calculate_prefill_costs(prefill_df, node_path, worker_nodes)
        # print("Prefill Communication Costs:", prefill_comm_costs)
        # print("Prefill Computation Costs:", prefill_comp_costs)

        # 2-5. Calculate decode costs
        decode_comm_costs, decode_comp_costs = calculate_decode_costs(decode_df, node_path, worker_nodes)
        
        # Calculate average values for communication and computation costs
        avg_decode_comm_costs = {key: (sum(values) / len(values) if values else 0) for key, values in decode_comm_costs.items()}
        avg_decode_comp_costs = {key: (sum(values) / len(values) if values else 0) for key, values in decode_comp_costs.items()}

        print("Decode Communication Costs (Average):", avg_decode_comm_costs)
        print("Decode Computation Costs (Average):", avg_decode_comp_costs)


if __name__ == "__main__":
    main()