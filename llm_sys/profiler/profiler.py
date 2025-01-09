# 2025.01.09 Lee JiHyuk
import time
import socket
from typing import List, Tuple, Dict

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
    def __init__(self, node_type: str, ip_address: str, port: int, duration: int):
        self.node_type: str = node_type
        self.ip_address: str = ip_address
        self.port: int = port
        self.repetitions = 100
        self.duration = duration

    def _send_packet(self, conn, packet: str):
        conn.sendall(packet.encode('utf-8'))

    def _receive_packet(self, conn) -> str:
        return conn.recv(1024).decode('utf-8')

class MasterProfiler(Profiler):
    """
    Master Profiler class.
    

    Args:
        slaves (List[Tuple[int, str]]): List for the (machine_id, ip_addresss)
        duration (int): executing time of system
    """
    
    def __init__(self, slaves: List[Tuple[int, str]], duration:int):
        super().__init__("master", ip_address=None, port=None, duration=duration)
        print("[Profiler] MasterProfiler Init!")
        
        # compute_node_index -> delta_time 
        self.delta_times: Dict[int, float] = {}
        # (compute_node_index, ip_address, port)
        self.slaves: List[Tuple[int, str]] = slaves
        
        time.sleep(1)

    def _handle_rtt_master(self, client_socket) -> float:
        """
        Measure RTT between master node and slave node.
        """
        
        rtts = []
        
        for _ in range(self.repetitions):
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
                self.delta_times[compute_node_idx] = (slave_ip, delta_time)
        
        # print delta_times
        print(self.delta_times)
        print("[Profiler] Finished delta_times measuring!")


class SlaveProfiler(Profiler):
    """
    Slave Profiler class.
    

    Args:
        ip_address (str): IP address of the machine
        port (int): port of the machine. 
                    Basic default = 9001
        duration (int): executing time of system
    """
    
    def __init__(self, duration: int, ip_address: str, port: int = 9001):
        super().__init__("slave", ip_address=ip_address, port=port, duration=duration)
        print("[Profiler] SlaveProfiler Init!")
        
        self.delta_time: float = 0.0

    def _handle_rtt_slave(self, conn) -> None:
        """
        Handle incoming RTT requests from the master.
        """
        
        for _ in range(self.repetitions):
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
