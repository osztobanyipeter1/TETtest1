import socket
import json

HOST = '192.168.249.52' # AR glasses IP address
PORT = 12345

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected to AR glasses server")
        
        while True:
            data = s.recv(1024).decode()
            if not data:
                break

            for line in data.split('\n'):
                if line.strip():
                    try:
                        orientation = json.loads(line)
                        print(f"Roll: {orientation['roll']:.2f}, Pitch: {orientation['pitch']:.2f}, Yaw: {orientation['yaw']:.2f}")
                    except json.JSONDecodeError:
                        print("Invalid data received:", line)

if __name__ == "__main__":
    main()