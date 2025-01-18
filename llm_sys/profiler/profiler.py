# 2025.01.09 Lee JiHyuk
import os
import time
import socket
from typing import List, Tuple, Dict
import csv
import pandas as pd
import re
from collections import defaultdict

def get_device_ip_configs(real_sys_config_file_name: str) -> List[Tuple[int, str, int]]:
    """
    Extract machine_id and ip_address from the given config file.

    Args:
        config_file (str): Path to the config.txt file.

    Returns:
        List[Tuple[int, str, int]]: List of (machine_id, ip_address, open_port for profiling communication) tuples.
    """
    machine_info = []

    with open(real_sys_config_file_name, 'r') as file:
        lines = file.readlines()

    machine_id = None
    ip_address = None
    open_port = None

    for line in lines:
        line = line.strip()
        if line.startswith("machine_id:"):
            machine_id = int(line.split(":")[1].strip())
        elif line.startswith("ip_address:"):
            ip_address = line.split(":")[1].strip()
        elif line.startswith("ports:"):
            ports = line.split(":")[1].strip().split()
            open_port = int(ports[-1])
        
        # When both machine_id and ip_address are found, add to the list
        if machine_id is not None and ip_address is not None and open_port is not None:
            machine_info.append((machine_id, ip_address, open_port))
            machine_id = None  # Reset for the next block
            ip_address = None
            open_port = None
    
    # Information about master node
    del machine_info[0]
    
    return machine_info

class Profiler:
    """
    Base Profiler class for default communication function
    """
    def __init__(self, file_directory: str, node_type: str, ip_address: str, port: int):
        self.node_type: str = node_type
        self.ip_address: str = ip_address
        self.port: int = port
        
        # For basic packet size
        self._packet_size: int = 1024
        # For delta_time measurement
        self._rtt_repeat: int = 1000
        # For event recording
        self._events: List[Tuple] = []
        # For collective file directory
        self._file_directory: str = file_directory

    def _send_packet(self, conn, packet: str):
        conn.sendall(packet.encode('utf-8'))

    def _receive_packet(self, conn) -> str:
        return conn.recv(self._packet_size).decode('utf-8')
    
    def _receive_file(self, conn, file_path) -> None:
        with open(file_path, "wb") as file:
            while True:
                data = conn.recv(self._packet_size)
                if not data:
                    break
                file.write(data)
        
        # print(f"[Profiler] File saved as {file_path}")
    
    def _send_file(self, conn, file_path) -> None:
        with open(file_path, "rb") as file:
            while (data := file.read(self._packet_size)):
                conn.sendall(data)
        
        print(f"[Profiler] File {file_path} sent successfully!")
        
    def connect_with_retry(self, client_socket, slave_ip, slave_port, max_retries=5):
        """
        Attempt to connect to a server with retries if the initial connection fails.

        Args:
            client_socket (socket.socket): The client socket.
            slave_ip (str): The IP address of the server.
            slave_port (int): The port of the server.
            max_retries (int): Maximum number of retries. Defaults to 5.

        Returns:
            bool: True if connection was successful, False if all retries failed.
        """
        retries = 0
        while retries < max_retries:
            try:
                print(f"[Profiler] Attempting to connect to {slave_ip}:{slave_port} (Try {retries + 1}/{max_retries})")
                client_socket.connect((slave_ip, slave_port))
                return True
            except (socket.error, ConnectionRefusedError) as e:
                retries += 1
                if retries < max_retries:
                    print(f"[Profiler] Connection failed: {e} | Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("[Profiler] All retries failed. Exiting.")
                    return False
    
    def record_event(self, time_stamp, request_id, in_out, mode, context_len, this_iter_processed):
        self._events.append((time_stamp, request_id, in_out, mode, context_len, this_iter_processed))
        
    def write_event_to_csv(self):
        """
        Write the events stored in `_events` to a CSV file in the desired format.

        Args:
            file_path (str): The path to the CSV file where events will be written.
        """
        assert self._file_path, f"There is no file path for events"
        with open(self._file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write a header row
            writer.writerow(["time_stamp", "request_id", "in_out", "mode", "context_len", "this_iter_processed"])
            writer.writerows(self._events)
            
            
class MasterProfiler(Profiler):
    """
    Master Profiler class.
    
    Works as client during communication functions
    
    Args:
        slaves (List[Tuple[int, str]]): List for the (machine_id, ip_addresss)
        duration (int): executing time of system
    """
    
    def __init__(self, slaves: List[Tuple[int, str]], duration:int, file_directory: str):
        super().__init__(node_type="master", file_directory=file_directory, ip_address=None, port=None)
        print("[Profiler] MasterProfiler Init!")
        
        # For duration
        self.duration = duration
        # compute_node_index -> delta_time 
        self.delta_times: Dict[int, Tuple[str, float]] = {}
        # (compute_node_index, ip_address, port)
        self.slaves: List[Tuple[int, str, int]] = slaves
        # For profiling the cluster's execution flow
        # (SRC, DST)
        self._node_path: List[Tuple[int, int]] = None
        # Worker nodes of the cluster
        self._worker_node: List[int] = []
        # For master event file path
        self._file_path: str = os.path.join(self._file_directory, "node_0_events.csv")
        
        time.sleep(1)

    def _handle_rtt_master(self, client_socket) -> float:
        """
        Measure RTT between master node and slave node.
        """
        
        rtts = []
        rtt_cnt = 0
        for _ in range(self._rtt_repeat):
            if rtt_cnt % 20 == 0:
                print(f"RTT Handling | {rtt_cnt} times done.")
                
            # 1-1. Get rtt start time
            start_time = time.time()
            
            # 1-2. Send rtt packet
            self._send_packet(client_socket, "PING")
            
            # 1-3. Receive rtt packet
            response = self._receive_packet(client_socket)
            
            # 1-4. Get rtt end time and return rtt
            end_time = time.time()
            if response == "PONG":
                rtts.append(end_time - start_time)
            else:
                raise ValueError("[Profiler] Unexpected response from slave.")
            
            rtt_cnt += 1
        
        return sum(rtts) / len(rtts)
        
    def _handle_delta_time_master(self, client_socket, rtt: float) -> float:
        """
        Send current time and RTT to the slave node.
        Finally, receive delta_time with slave node
        """
        
        # 2-1. Get master_time
        now = time.time()
        
        # 2-2. Send master_time&rtt
        self._send_packet(client_socket, f"SYNC, {now}, {rtt}")
        
        # 2-3. Wait and receive delta_time 
        # SYNC, delta_time
        data = self._receive_packet(client_socket)

        if data.startswith("SYNC"):
            _, delta_time = data.split(",")
            delta_time = float(delta_time)
            
        return delta_time

    def start_master_profiling(self) -> None:
        """
        Start profiling by measuring RTT and synchronizing time with slaves.
        """
        
        for compute_node_idx, slave_ip, slave_port in self.slaves:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                # 0. Connect to the slave node
                print(f"[Profiler] Master connecting to compute node {compute_node_idx}({slave_ip}:{slave_port})")
                self.connect_with_retry(client_socket, slave_ip, slave_port, max_retries=100)
                
                # 1. Measure RTT between slave node
                rtt = self._handle_rtt_master(client_socket)
                
                # 2. Measure delta_time with slave node
                delta_time = self._handle_delta_time_master(client_socket, rtt)
                self.delta_times[compute_node_idx] = (slave_ip, float(delta_time))
                
                # 3. Send duration to the slave nodes
                self._send_packet(client_socket, f"{self.duration}")
        
        # print delta_times
        print(self.delta_times)
        print("[Profiler] Finished delta_times measuring!")
    
    def _collect_events(self) -> None:
        for compute_node_idx, slave_ip, slave_port in self.slaves:
            self._worker_node.append(compute_node_idx)
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:       
                # 0. Connect to the slave node
                print(f"[Profiler] Master connecting to compute node {compute_node_idx}({slave_ip}:{slave_port})")
                self.connect_with_retry(client_socket, slave_ip, slave_port, max_retries=100)
                
                # 1. Receive events file from the slave node
                file_name = f"node_{compute_node_idx}_events.csv"
                event_file_path = os.path.join(self._file_directory, file_name)
                self._receive_file(conn=client_socket, file_path=event_file_path)
                
    def _merge_events(self) -> str:
        # 1. Get event file list
        file_list = [file for file in os.listdir(self._file_directory) 
                        if file.endswith("_events.csv") and os.path.isfile(os.path.join(self._file_directory, file))]
        
        # 2. Merge event files into a file
        dataframes = []
        for file in file_list:
            # 2-1. Find node name from file
            match = re.match(r"node_(.*?)_events\.csv", file)
            if match:
                node_name = match.group(1)
            else:
                print(f"[Profiler] Skipping file: {file} (node_name not found)")
                continue
            
            # 2-2. Read file 
            file_path = os.path.join(self._file_directory, file)
            df = pd.read_csv(file_path)
            
            # 2-3. Add source info. and apply delta time
            df['source'] = str(node_name)
            df['time_stamp'] = pd.to_numeric(df['time_stamp'], errors='coerce')
            if node_name != '0':
                delta_time = self.delta_times[int(node_name)][1]
                df['time_stamp'] -= delta_time
            
            # 2-4. Concat file
            dataframes.append(df)
            
        # 3. Sort by time_stamp
        combined_df = pd.concat(dataframes, ignore_index=True)
        sorted_df = combined_df.sort_values(by='time_stamp').reset_index(drop=True)
        
        # 4. Save sorted DataFrame into new csv file
        output_file = os.path.join(self._file_directory, "merged_sorted.csv")
        sorted_df.to_csv(output_file, index=False)
        print(f"[Profiler] Merged and sorted CSV saved to: {output_file}")
        
        return output_file
    
    def _calculate_prefill_costs(self, prefill_df):
        """
        Process a single request_id to calculate filtered_df, decode_comm_costs, and decode_comp_costs.
        
        :param prefill_df: The prefill DataFrame by request_id
        :return: Tuple of (decode_comm_costs, decode_comp_costs)
        """
        
        # Communication cost: (src, dst) -> List
        prefill_comm_costs: Dict[Tuple[int, int], float] = defaultdict(float)
        # Computation cost: node_index -> List
        prefill_comp_costs: Dict[int, float] = defaultdict(float)
        
        for src, dst in self._node_path:
            src_cond = (prefill_df['in_out'] == 'out') & (prefill_df['source'] == src)
            dst_cond = (prefill_df['in_out'] == 'in') & (prefill_df['source'] == dst)

            src_row = prefill_df[src_cond]
            dst_row = prefill_df[dst_cond]

            if not src_row.empty and not dst_row.empty:
                prefill_comm_costs[(src, dst)] = dst_row.iloc[0]['time_stamp'] - src_row.iloc[0]['time_stamp']
        
        for node in self._worker_node:
            in_cond = (prefill_df['in_out'] == 'in') & (prefill_df['source'] == node)
            out_cond = (prefill_df['in_out'] == 'out') & (prefill_df['source'] == node)

            in_row = prefill_df[in_cond]
            out_row = prefill_df[out_cond]

            if not in_row.empty and not out_row.empty:
                prefill_comp_costs[node] = out_row.iloc[0]['time_stamp'] - in_row.iloc[0]['time_stamp']
            
        return prefill_comm_costs, prefill_comp_costs
        
    def _calculate_decode_costs(self, decode_df):
        """
        Process a single request_id to calculate filtered_df, decode_comm_costs, and decode_comp_costs.
        
        :param decode_df: The decode DataFrame by request_id
        :return: Tuple of (decode_comm_costs, decode_comp_costs)
        """
        
        # Communication cost: (src, dst) -> List
        decode_comm_costs: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        # Computation cost: node_index -> List
        decode_comp_costs: Dict[int, List[float]] = defaultdict(list)
        
        for src, dst in self._node_path:
            src_cond = (decode_df['in_out'] == 'out') & (decode_df['source'] == src)
            dst_cond = (decode_df['in_out'] == 'in') & (decode_df['source'] == dst)

            src_rows = decode_df[src_cond].reset_index(drop=True)
            dst_rows = decode_df[dst_cond].reset_index(drop=True)

            min_len = min(len(src_rows), len(dst_rows))
            for i in range(min_len):
                src_time = src_rows.loc[i, 'time_stamp']
                dst_time = dst_rows.loc[i, 'time_stamp']
                decode_comm_costs[(src, dst)].append(dst_time - src_time)
        
        for node in self._worker_node:
            in_cond = (decode_df['in_out'] == 'in') & (decode_df['source'] == node)
            out_cond = (decode_df['in_out'] == 'out') & (decode_df['source'] == node)

            in_rows = decode_df[in_cond].reset_index(drop=True)
            out_rows = decode_df[out_cond].reset_index(drop=True)

            # Ensure in_rows and out_rows have the same length for 1:1 mapping
            min_len = min(len(in_rows), len(out_rows))
            if len(in_rows) == len(out_rows):
                for i in range(min_len):
                    in_time = in_rows.loc[i, 'time_stamp']
                    out_time = out_rows.loc[i, 'time_stamp']
                    decode_comp_costs[node].append(out_time - in_time)
            else:
                for i in range(1, min_len - 1):
                    in_time = in_rows.loc[i, 'time_stamp']
                    out_time = out_rows.loc[i + 1, 'time_stamp']
                    
                    if out_time - in_time > 0:
                        decode_comp_costs[node].append(out_time - in_time)
        return decode_comm_costs, decode_comp_costs
    
    def _calculate_delays(self, file_path) -> None:
        # 1. Read sorted file from decode stage
        merged_df = pd.read_csv(file_path)
        
        # When the later moment come, node_path should be integrated within total codes
        # At now, node_path is not ours interests  
        self._node_path = [(0, 1), (1, 2), (2, 0)]
        
        # comm: (src, dst) -> List[comm_delay]
        # comp: node -> List[comp_delay]
        prefill_comm_costs : Dict[Tuple[int, int], List[float]] = defaultdict(list)
        prefill_comp_costs : Dict[int, List[float]] = defaultdict(list)
        decode_comm_costs : Dict[Tuple[int, int], List[float]] = defaultdict(list)
        decode_comp_costs : Dict[int, List[float]] = defaultdict(list)
        
        # 2. Iterate through every request_ids
        request_id_list = merged_df['request_id'].unique().tolist()
        for request_id in request_id_list:
            # 2-1. Filter merged_df by request_id
            request_condition = merged_df["request_id"] == request_id
            request_df = merged_df[request_condition].reset_index(drop=True)
            
            if request_df is None:
                print(f"Request ID {request_id} has no filtered DataFrame")
                continue
            
            # 2-2. Find criterion line
            decode_start_condition = (request_df['in_out'] == 'out') & \
                                    (request_df['mode'] == 'decode') & \
                                    (request_df['source'] == 0)
            decode_start_index = request_df.index[decode_start_condition].min()

            # 2-3. Devide Dataframe by prefill and decode
            if pd.notna(decode_start_index):
                prefill_df = request_df.iloc[:decode_start_index].reset_index(drop=True)
                decode_df = request_df.iloc[decode_start_index:].reset_index(drop=True)
            else:
                print(f"No rows satisfy the start condition for request_id {request_id}.")
                continue
            
            # 2-4. Calculate prefill and decode delay
            req_prefill_comm_costs, req_prefill_comp_costs = self._calculate_prefill_costs(prefill_df=prefill_df)
            req_decode_comm_costs, req_decode_comp_costs = self._calculate_decode_costs(decode_df=decode_df)
            
            for src, dst in self._node_path:
                prefill_comm_costs[(src, dst)].append(req_prefill_comm_costs[(src, dst)])
                decode_comm_costs[(src, dst)].extend(req_decode_comm_costs[(src, dst)])
            
            for node in self._worker_node:
                prefill_comp_costs[node].append(req_prefill_comp_costs[node])
                decode_comp_costs[node].extend(req_decode_comp_costs[node])
        
        # 3. Save the results to csv files
        comm_data = []
        for (src, dst), prefill_list in prefill_comm_costs.items():
            decode_list = decode_comm_costs.get((src, dst), [])
            comm_data.append({
                '(src, dst)': (src, dst),
                'Prefill_comm': sum(prefill_list) / len(prefill_list) if prefill_list else 0,
                'Decode_comm': sum(decode_list) / len(decode_list) if decode_list else 0
            })
        comm_df = pd.DataFrame(comm_data)
        
        comp_data = []
        for node, prefill_list in prefill_comp_costs.items():
            decode_list = decode_comp_costs.get(node, [])
            comp_data.append({
                'node': node,
                'Prefill_comp': sum(prefill_list) / len(prefill_list) if prefill_list else 0,
                'Decode_comp': sum(decode_list) / len(decode_list) if decode_list else 0
            })
        comp_df = pd.DataFrame(comp_data)
        
        # Save to separate CSV files
        comm_output_file = os.path.join(self._file_directory, "comm.csv")
        comp_output_file = os.path.join(self._file_directory, "comp.csv")

        comm_df.to_csv(comm_output_file, index=False, quoting=1)
        comp_df.to_csv(comp_output_file, index=False, quoting=1)

        print(f"Communication costs saved to {comm_output_file}")
        print(f"Computation costs saved to {comp_output_file}")
        
        # 4. Print the results
        print("\n[Profiler] Profiling final result")
        print(f"{'Type':<15}{'Source/Destination':<25}{'Prefill Cost':<20}{'Decode Cost':<20}")
        print("-" * 80)

        # Print communication costs
        for (src, dst), prefill_list in prefill_comm_costs.items():
            decode_list = decode_comm_costs.get((src, dst), [])
            prefill_avg = sum(prefill_list) / len(prefill_list) if prefill_list else 0
            decode_avg = sum(decode_list) / len(decode_list) if decode_list else 0
            print(f"{'Comm':<15}{str((src, dst)):<25}{prefill_avg:<20.6f}{decode_avg:<20.6f}")

        # Print computation costs
        for node, prefill_list in prefill_comp_costs.items():
            decode_list = decode_comp_costs.get(node, [])
            prefill_avg = sum(prefill_list) / len(prefill_list) if prefill_list else 0
            decode_avg = sum(decode_list) / len(decode_list) if decode_list else 0
            print(f"{'Comp':<15}{node:<25}{prefill_avg:<20.6f}{decode_avg:<20.6f}")
    
    def generate_delay_report(self) -> None:
        # 1. Collect event files from compute nodes
        self._collect_events()
        
        # 2. Merge event files
        merged_file_path = self._merge_events()
        
        # 3. Calculate delays from merged file
        self._calculate_delays(merged_file_path)

class SlaveProfiler(Profiler):
    """
    Slave Profiler class.
    
    Works as server during communication funtions
    
    Args:
        ip_address (str): IP address of the machine
        port (int): port of the machine. 
                    Basic default = 9001
        duration (int): executing time of system
    """
    
    def __init__(self, file_directory: str, ip_address: str, port: int = 9001):
        super().__init__(node_type="slave", file_directory=file_directory, ip_address="0.0.0.0", port=port)
        print("[Profiler] SlaveProfiler Init!")
        
        # For delta time value
        self.delta_time: float = 0.0
        # For event file path
        self._file_path: str = os.path.join(self._file_directory, "events.csv")
        # For duration
        self._duration: int = None

    def _handle_rtt_slave(self, conn) -> None:
        """
        Handle incoming RTT requests from the master.
        """
        
        for _ in range(self._rtt_repeat):
            # 2-1. Wait and receive rtt packet
            data = self._receive_packet(conn)
            
            # 2-2. Check and send rtt packet
            if data == "PING":
                self._send_packet(conn, "PONG")  # Respond to RTT measurement
            else:
                RuntimeError("[Profiler] Wrong packet")
            
    def _handle_delta_time_slave(self, conn) -> None:
        """
        Handle incoming delta_time requests from the master
        """
        
        # 3-1. Wait and receive master_time&rtt
        # SYNC, master_time, rtt
        data = self._receive_packet(conn)
        
        # 3-2. Check received packet
        if data.startswith("SYNC"):
            _, master_time, rtt = data.split(",")
            master_time = float(master_time)
            rtt = float(rtt)
            
            self.delta_time = time.time() - (master_time + rtt/2)
            print("[Profiler] delta_time:", self.delta_time)
            
            # 3-3. Send delta_time
            self._send_packet(conn, f"SYNC, {self.delta_time}")
            
    def get_duration(self) -> int:
        return self._duration

    def start_slave_profiling(self):
        """
        Start the slave node to listen for master requests.
        """
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # 0. Await until master node connects
            server_socket.bind((self.ip_address, self.port))
            server_socket.listen()
            conn, addr = server_socket.accept()
            
            # 1. Complete connection between master node
            with conn:
                print(f"[Profiler] Connection established with control node({addr})")
                
                # 2. Measure RTT between master node
                self._handle_rtt_slave(conn)
                
                # 3. Measure delta_time with master node
                self._handle_delta_time_slave(conn)   
                
                # 4. Receive duration
                duration = self._receive_packet(conn)
                self._duration = int(duration)
            
            # 4. Exit the connection with master
            print("[Profiler] Finished delta_time measuring!")

    def send_delay_report(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # 0. Await until master node connects
            server_socket.bind((self.ip_address, self.port))
            server_socket.listen()
            conn, addr = server_socket.accept()
            
            # 1. Send events file to the master node
            with conn:
                print(f"[Profiler] Connection established with control node({addr})")
                
                self._send_file(conn=conn, file_path=self._file_path)
    
# Example Usage
if __name__ == "__main__":
    # Example to run master or slave
    import sys
    
    if sys.argv[1] == "master":
        slaves = [(0, "127.0.0.1")]  # List of slave nodes (IP, Port)
        master = MasterProfiler(slaves)
        master.start_master_profiling()
        print("Delta times with slaves:", master.delta_times)

    elif sys.argv[1] == "slave":
        slave = SlaveProfiler(ip_address="127.0.0.1", port=9001)
        slave.start_slave_profiling()
