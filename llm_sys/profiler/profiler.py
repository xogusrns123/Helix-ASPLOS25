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
        self._repetitions: int = 100
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
        # For master event file path
        self._file_path: str = os.path.join(self._file_directory, "node_master_events.csv")
        
        time.sleep(1)

    def _handle_rtt_master(self, client_socket) -> float:
        """
        Measure RTT between master node and slave node.
        """
        
        rtts = []
        rtt_cnt = 0
        for _ in range(self._repetitions):
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
            if node_name != 'master':
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
    
    def _calculate_decode_costs(self, filtered_df):
        """
        Process a single request_id to calculate filtered_df, decode_comm_costs, and decode_comp_costs.
        
        :param merged_df: The full merged DataFrame
        :param request_id: The request ID to process
        :return: Tuple of (filtered_df, decode_comm_costs, decode_comp_costs)
        """
        # 2-1. Initialize data structure
        # Computation cost: node_index -> List
        decode_comp_costs: Dict[int, List] = defaultdict(list)
        # Communication cost: (src, dst) -> List
        decode_comm_costs: Dict[Tuple[int, int], List] = defaultdict(list)
    
        # start_index = first line where the decoding stage begins
        decode_start_condition = (filtered_df['in_out'] == 'in') & \
                        (filtered_df['mode'] == 'prompt') & \
                        (filtered_df['source'] == 'master')
        decode_start_index = filtered_df.index[decode_start_condition].min()

        if pd.notna(decode_start_index):
            decode_df = filtered_df.iloc[decode_start_index:].reset_index(drop=True)
        else:
            print(f"No rows satisfy the start condition for request_id {filtered_df.iloc[3]['request_id']}.")
            return None, None, None

        # 2-2. Iterate through the rows to calculate costs
        for i in range(1, len(decode_df)):
            prev_row = decode_df.iloc[i - 1]
            curr_row = decode_df.iloc[i]
            
            # 2-2-1. Calculate time difference
            time_diff = curr_row['time_stamp'] - prev_row['time_stamp']
            src_node = prev_row['source']
            dst_node = curr_row['source']
            
            # 2-2-2. Communication cost: specific transitions
            if (src_node != dst_node):
                decode_comm_costs[(src_node, dst_node)].append(time_diff)
            # 2-2-3. Computation cost: same source
            elif (src_node == dst_node and src_node != 'master'):
                decode_comp_costs[src_node].append(time_diff)

        return decode_comm_costs, decode_comp_costs
    
    def _calculate_delays(self, file_path) -> None:
        # 1. Read sorted file from decode stage
        merged_df = pd.read_csv(file_path)
        
        # Total computation cost: request_id -> average computation cost
        total_comp_cost: Dict[int, float] = {}
        # Total communication cost: request_id -> average communication cost
        total_comm_cost: Dict[int, float] = {}
        
        # 2. Iterate through every request_ids
        request_id_list = merged_df['request_id'].unique().tolist()
        for request_id in request_id_list:
            filtered_condition = merged_df["request_id"] == request_id
            filtered_df = merged_df[filtered_condition].reset_index(drop=True)
            
            if filtered_df is None:
                print(f"Request ID {request_id} has no filtered DataFrame")
                continue
    
            decode_comm_costs, decode_comp_costs = self._calculate_decode_costs(filtered_df)
            
            # 2-3. Calculate averages
            total_comm_sum = sum(sum(times) for times in decode_comm_costs.values())
            total_comm_len = sum(len(times) for times in decode_comm_costs.values())
            total_comp_sum = sum(sum(times) for times in decode_comp_costs.values())
            total_comp_len = sum(len(times) for times in decode_comp_costs.values())
                
            average_communication_cost = total_comm_sum / total_comm_len if decode_comm_costs else 0
            average_computation_cost = total_comp_sum / total_comp_len if decode_comp_costs else 0
            
            total_comm_cost[request_id] = average_communication_cost
            total_comp_cost[request_id] = average_computation_cost

        # 3. Print the results
        print(f"[Profiler] Profiling final result")
        print(f"{'Request ID':<15}{'Computation Cost':<20}{'Communication Cost':<20}")
        print("-" * 55)

        # Iterate through all request IDs
        for req_id in sorted(total_comp_cost.keys() | total_comm_cost.keys()):
            # Get computation and communication costs, defaulting to 0
            comp_cost = total_comp_cost.get(req_id, 0)
            comm_cost = total_comm_cost.get(req_id, 0)

            # Fixed-width formatting for rows
            print(f"{req_id:<15}{comp_cost:<20.6f}{comm_cost:<20.6f}")
        print()
    
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
        
        for _ in range(self._repetitions):
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
