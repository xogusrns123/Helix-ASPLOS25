# 2025.01.09 Lee JiHyuk
import os
import time
import socket
from typing import List, Tuple, Dict
import csv
import pandas as pd
import re

def get_device_ip_configs(real_sys_config_file_name: str) -> List[Tuple[int, str]]:
    """
    Extract machine_id and ip_address from the given config file.

    Args:
        config_file (str): Path to the config.txt file.

    Returns:
        List[Tuple[int, str]]: List of (machine_id, ip_address) tuples.
    """
    machine_info = []

    with open(real_sys_config_file_name, 'r') as file:
        lines = file.readlines()

    machine_id = None
    ip_address = None

    for line in lines:
        line = line.strip()
        if line.startswith("machine_id:"):
            machine_id = int(line.split(":")[1].strip())
        elif line.startswith("ip_address:"):
            ip_address = line.split(":")[1].strip()
        
        # When both machine_id and ip_address are found, add to the list
        if machine_id is not None and ip_address is not None:
            machine_info.append((machine_id, ip_address))
            machine_id = None  # Reset for the next block
            ip_address = None
    
    # Information about master node
    del machine_info[0]
    
    return machine_info

class Profiler:
    """
    Base Profiler class for default communication function
    """
    def __init__(self, duration: int, file_directory: str, node_type: str, ip_address: str, port: int):
        self.node_type: str = node_type
        self.ip_address: str = ip_address
        self.port: int = port
        self.duration: int = duration
        
        # For basic packet size
        self._packet_size: int = 1024
        # For delta_time measurement
        self._repetitions: int = 1000
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
        
        print(f"[Profiler] File saved as {file_path}")
    
    def _send_file(self, conn, file_path) -> None:
        with open(file_path, "rb") as file:
            while (data := file.read(self._packet_size)):
                conn.sendall(data)
        
        print(f"[Profiler] File {file_path} sent successfully!")
    
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
        super().__init__(node_type="master", duration=duration, file_directory=file_directory, ip_address=None, port=None)
        print("[Profiler] MasterProfiler Init!")
        
        # compute_node_index -> delta_time 
        self.delta_times: Dict[int, Tuple[str, float]] = {}
        # (compute_node_index, ip_address, port)
        self.slaves: List[Tuple[int, str]] = slaves
        # For master event file path
        self._file_path: str = os.path.join(self._file_directory, "node_master_events.csv")
        
        time.sleep(1)

    def _handle_rtt_master(self, client_socket) -> float:
        """
        Measure RTT between master node and slave node.
        """
        
        rtts = []
        
        for _ in range(self._repetitions):
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
        
        for compute_node_idx, slave_ip in self.slaves:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:        
                slave_port = 9001
                
                # 0. Connect to the slave node
                print(f"[Profiler] Master connecting to compute node {compute_node_idx}({slave_ip}:{slave_port})")
                client_socket.connect((slave_ip, slave_port))
                
                # 1. Measure RTT between slave node
                rtt = self._handle_rtt_master(client_socket)
                
                # 2. Measure delta_time with slave node
                delta_time = self._handle_delta_time_master(client_socket, rtt)
                self.delta_times[compute_node_idx] = (slave_ip, float(delta_time))
        
        # print delta_times
        print(self.delta_times)
        print("[Profiler] Finished delta_times measuring!")
    
    def _collect_events(self) -> None:
        for compute_node_idx, slave_ip in self.slaves:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:       
                # 0. Connect to the slave node
                slave_port = 9001
                print(f"[Profiler] Master connecting to compute node {compute_node_idx}({slave_ip}:{slave_port})")
                client_socket.connect((slave_ip, slave_port))
                
                # 1. Receive events file from the slave node
                file_name = f"node_{compute_node_idx}_events.csv"
                event_file_path = os.path.join(self._file_directory, file_name)
                self._receive_file(conn=client_socket, file_path=event_file_path)
                
        print("[Profiler] Every files have been saved successfully.")
                
    def _merge_events(self) -> None:
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
            if node_name != "master":
                delta_time = self.delta_times[int(node_name)]
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
    
    def _calculate_delays(self) -> None:
        # Not yet implemented
        pass
    
    def generate_delay_report(self) -> None:
        # 1. Collect event files from compute nodes
        self._collect_events()
        
        # 2. Merge event files
        self._merge_events()
        
        # 3. Calculate delays from merged file
        self._calculate_delays()

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
    
    def __init__(self, duration: int, file_directory: str, ip_address: str, port: int = 9001):
        super().__init__(node_type="slave", duration=duration, file_directory=file_directory, ip_address=ip_address, port=port)
        print("[Profiler] SlaveProfiler Init!")
        
        # For delta time value
        self.delta_time: float = 0.0
        # For event file path
        self._file_path: str = os.path.join(self._file_directory, "events.csv")

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
                
        print("[Profiler] File sent successfully.")
    
    
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
