import socket
import json
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class PoseVisualizer:
    def __init__(self, host='0.0.0.0', port=12346):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(1)
        print(f"Listening on {host}:{port}")

        self.position = np.zeros(3)
        self.orientation = np.array([1,0,0,0])  # quaternion w,x,y,z
        self.camera_front = np.array([0,0,-1])
        self.client_socket = None
        self.running = True

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        thread = threading.Thread(target=self.accept_client)
        thread.daemon = True
        thread.start()

    def accept_client(self):
        self.client_socket, _ = self.sock.accept()
        print("Client connected")

        buffer = ""
        while self.running:
            data = self.client_socket.recv(1024).decode()
            if not data:
                break
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    pose_data = json.loads(line)
                    self.position = np.array(pose_data['position'])
                    self.orientation = np.array([
                        pose_data['orientation']['w'],
                        pose_data['orientation']['x'],
                        pose_data['orientation']['y'],
                        pose_data['orientation']['z']
                    ])
                    self.camera_front = np.array(pose_data['camera_front'])
                except:
                    pass

    def update_plot(self, frame):
        self.ax.clear()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('AR Position and Orientation')

        # Plot camera position
        self.ax.scatter(*self.position, c='r', label='Position')

        # Plot camera front direction as arrow
        front = self.camera_front / np.linalg.norm(self.camera_front)
        self.ax.quiver(
            self.position[0], self.position[1], self.position[2],
            front[0], front[1], front[2],
            length=2.0, color='b', label='Orientation'
        )

        self.ax.legend()

    def run(self):
        import matplotlib.animation as animation
        ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        self.sock.close()

if __name__ == "__main__":
    visualizer = PoseVisualizer()
    visualizer.run()
