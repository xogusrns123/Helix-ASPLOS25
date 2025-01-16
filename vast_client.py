import socket

def get_used_ports(server_ip, server_port, connection_count):
    """
    Connects to the server multiple times and prints the local ports used for each connection.

    :param server_ip: IP address of the server to connect to.
    :param server_port: Port number of the server.
    :param connection_count: Number of connections to establish.
    """
    client_sockets = []

    for i in range(connection_count):
        try:
            # Create a socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Allow the OS to assign a random local port
            client_socket.bind(('', 0))  # Bind to any available local port
            
            client_socket.connect((server_ip, server_port))

            # Get the local socket information
            local_address = client_socket.getsockname()
            print(f"Connection {i + 1}: Local IP = {local_address[0]}, Port = {local_address[1]}")

            client_sockets.append(client_socket)
        except Exception as e:
            print(f"Error in connection {i + 1}: {e}")

    # Close all sockets
    for client_socket in client_sockets:
        client_socket.close()

if __name__ == "__main__":
    SERVER_IP = "27.65.59.245"  # Master node IP
    SERVER_PORT = 54417           # Master node port
    CONNECTION_COUNT = 5          # Number of connections to establish

    print(f"Connecting to server {SERVER_IP}:{SERVER_PORT}...")
    get_used_ports(SERVER_IP, SERVER_PORT, CONNECTION_COUNT)
