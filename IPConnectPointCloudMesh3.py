import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import socket
import json
import time

class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=0.1**2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        kalman_gain = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate
        return self.posteri_estimate

WINDOW_SECONDS = 15
time_data = []
x_data, y_data, z_data = [], [], []
x_filtered, y_filtered, z_filtered = [], [], []
kalman_x = SimpleKalmanFilter()
kalman_y = SimpleKalmanFilter()
kalman_z = SimpleKalmanFilter()
start_time = time.time()

def data_thread():
    global x_data, y_data, z_data, x_filtered, y_filtered, z_filtered, time_data
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 12345))
    buffer = ""
    while True:
        try:
            data = s.recv(1024).decode()
            if not data:
                break
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    try:
                        d = json.loads(line)
                        t = time.time() - start_time
                        x, y, z = d['x'], d['y'], d['z']
                        time_data.append(t)
                        x_data.append(x)
                        y_data.append(y)
                        z_data.append(z)
                        x_filtered.append(kalman_x.update(x))
                        y_filtered.append(kalman_y.update(y))
                        z_filtered.append(kalman_z.update(z))
                    except Exception as e:
                        print("JSON error:", e)
        except Exception as e:
            print("Socket error:", e)
            break

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

def animate(i):
    if not time_data:
        return
    t_now = time_data[-1]
    t_min = t_now - WINDOW_SECONDS
    indices = [i for i, t in enumerate(time_data) if t >= t_min]

    ax1.clear()
    ax2.clear()
    ax3.clear()
    if not indices:
        return

    idx_start = indices[0]

    td = time_data[idx_start:]
    xd = x_data[idx_start:]
    yd = y_data[idx_start:]
    zd = z_data[idx_start:]
    xf = x_filtered[idx_start:]
    yf = y_filtered[idx_start:]
    zf = z_filtered[idx_start:]

    ax1.plot(td, xd, label='x (raw)', color='red')
    ax1.plot(td, xf, label='x (filtered)', color='green')
    ax1.legend()
    ax1.set_ylabel('X')

    ax2.plot(td, yd, label='y (raw)', color='blue')
    ax2.plot(td, yf, label='y (filtered)', color='green')
    ax2.legend()
    ax2.set_ylabel('Y')

    ax3.plot(td, zd, label='z (raw)', color='orange')
    ax3.plot(td, zf, label='z (filtered)', color='green')
    ax3.legend()
    ax3.set_ylabel('Z')
    ax3.set_xlabel('Time (s)')

data_recv_thread = threading.Thread(target=data_thread, daemon=True)
data_recv_thread.start()

ani = animation.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)
plt.tight_layout()
plt.show()
