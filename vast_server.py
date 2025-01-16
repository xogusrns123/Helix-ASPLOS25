import socket
import threading

def handle_client(client_socket, client_address):
    """Handle communication with a connected client."""
    print(f"New connection from {client_address[0]}:{client_address[1]}")

    try:
        while True:
            data = client_socket.recv(1024)  # Receive data from the client
            if not data:
                print(f"Connection closed by {client_address[0]}:{client_address[1]}")
                break

            print(f"Received from {client_address[0]}:{client_address[1]}: {data.decode()}")

            # Echo the data back to the client
            client_socket.sendall(data)
    except Exception as e:
        print(f"Error handling client {client_address[0]}:{client_address[1]}: {e}")
    finally:
        client_socket.close()
        print(f"Connection with {client_address[0]}:{client_address[1]} closed.")

def run_server(host, port):
    """Start the server and accept connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            client_handler = threading.Thread(
                target=handle_client, args=(client_socket, client_address)
            )
            client_handler.start()
    except KeyboardInterrupt:
        print("Shutting down the server.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    HOST = "0.0.0.0"  # Listen on all available network interfaces
    PORT = 42744       # Port to listen on

    run_server(HOST, PORT)
