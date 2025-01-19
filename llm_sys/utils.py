# 2024.04.24 Yixuan Mei
# 2025.01.16 Modified by Lee JiHyuk

import torch
import socket
import requests
import os

# in simulator, the first compute node has idx 2
# in real sys, the first compute node has idx 1
# simulator idx - SIMULATOR_NODE_OFFSET = real node idx
SIMULATOR_NODE_OFFSET = 1


def to_real(node_id: int) -> int:
    if node_id == 0 or node_id == 1:
        return 0
    else:
        return node_id - SIMULATOR_NODE_OFFSET


# CONFIG_BROADCAST_ADDR = "tcp://10.128.0.31:5000"
# CONFIG_BROADCAST_ADDR = "tcp://143.248.53.59:5000"
CONFIG_BROADCAST_ADDR = "tcp://143.248.53.100:9000"
HOST_CONFIG_BROADCAST_ADDR = "dummy"
WORKER_CONFIG_BROADCAST_ADDR = "tcp://34.47.67.23:7000"
VAST_AI = True

if VAST_AI:
    START_PORT = 70000
else:
    START_PORT = 6000

def warm_up():
    # create a tensor and move it to GPU (Warm up GPU)
    x = torch.tensor([1, 2, 3])
    for i in range(100):
        x.cuda()

def get_local_ip():
    # Attempt to connect to an internet host in order to determine the local interface
    try:
        # Create a dummy socket to connect to an Internet IP or DNS
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Use Google's public DNS server to find out our IP
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return ip
    except Exception as e:
        return f"Error obtaining local IP: {str(e)}"

def get_public_ip():
    try:
        return requests.get("http://ifconfig.me").text.strip()
    except Exception as e:
        return f"Error obtaining public IP: {str(e)}"

def make_self_config(device_num) -> tuple[str, int]:
    if VAST_AI:
        ip_address: str = get_public_ip()
    else:
        ip_address: str = get_local_ip()
    
    ports = ""
    for device_idx in range(device_num + 1):
        port = START_PORT + device_idx
        
        if VAST_AI:
            ENV_PORT = "VAST_TCP_PORT_" + str(port)
            port = int(os.environ.get(ENV_PORT))
            ports += f"{port} "
        else:
            ports += f"{port} "
    
    output_path = './config/device_config.txt'
    
    with open(output_path, "w") as file:
        file.write("ip_address: " + ip_address + "\n")
        file.write(f"ports: {ports}\n\n")
    
    return ip_address, port

class FlyingQuery:
    def __init__(self, query_uid, input_length, output_length, compute_node_uids, start_layers, end_layers, pipeline):
        self.query_uid = query_uid
        self.input_length = input_length
        self.output_length = output_length
        self.processed_tokens = 0

        # scheduling
        self.compute_node_uids = compute_node_uids
        self.start_layers = start_layers
        self.end_layers = end_layers
        # List[PipelineStage]
        self.pipeline = pipeline
