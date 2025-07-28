import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import threading
import socket
import json
import time


WINDOW_SECONDS = 15

# A két pozícióadatot párhuzamosan tároljuk
class PositionData:
    def __init__(self):
        self.lock = threading.Lock()

        # Kalman-szűrt pozíció
        self.time_k = []
        self.x_k = []
        self.y_k = []
        self.z_k = []

        # Raw pozíció
        self.time_r = []
        self.x_r = []
        self.y_r = []
        self.z_r = []

pos_data = PositionData()
start_time = time.time()


def data_thread_kalman():
    global pos_data
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 12345))
    buffer = ""
    while True:
        try:
            data = s.recv(1024).decode()
            if not data:
                break
            buffer += data
            while '\n' in buffer or '\r' in buffer:
                # Vegyük figyelembe akár \n vagy \r sortöréseket
                if '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                else:
                    line, buffer = buffer.split('\r',1)
                if line.strip():
                    try:
                        d = json.loads(line)
                        t = time.time() - start_time
                        with pos_data.lock:
                            pos_data.time_k.append(t)
                            pos_data.x_k.append(d['x'])
                            pos_data.y_k.append(d['y'])
                            pos_data.z_k.append(d['z'])
                    except Exception as e:
                        print("Kalman JSON error:", e)
        except Exception as e:
            print("Kalman socket error:", e)
            break


def data_thread_raw():
    global pos_data
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 12346))
    buffer = ""
    while True:
        try:
            data = s.recv(1024).decode()
            if not data:
                break
            buffer += data
            while '\n' in buffer or '\r' in buffer:
                if '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                else:
                    line, buffer = buffer.split('\r',1)
                if line.strip():
                    try:
                        d = json.loads(line)
                        t = time.time() - start_time
                        with pos_data.lock:
                            pos_data.time_r.append(t)
                            pos_data.x_r.append(d['x'])
                            pos_data.y_r.append(d['y'])
                            pos_data.z_r.append(d['z'])
                    except Exception as e:
                        print("Raw JSON error:", e)
        except Exception as e:
            print("Raw socket error:", e)
            break


# Indítsuk el a socket olvasó szálakat
threading.Thread(target=data_thread_kalman, daemon=True).start()
threading.Thread(target=data_thread_raw, daemon=True).start()


# Ábrázoló és animációs részek

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Pozíció követés (Kalman vs Raw)")

# Tengely beállítása (tetszés szerint módosítható)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel('X (Előre-Hátra)')
ax.set_ylabel('Y (Fel-Le)')
ax.set_zlabel('Z (Jobbra-Balra)')

line_kalman, = ax.plot([], [], [], 'o-', color='green', label='Kalman szűrt pozíció')
#line_raw, = ax.plot([], [], [], 'o-', color='red', label='Raw pozíció')

ax.legend()

def animate(i):
    with pos_data.lock:
        # Csak az elmúlt WINDOW_SECONDS adatokat mutatjuk
        # Kalman adatok szűrése
        if not pos_data.time_k:
            return
        t_now_k = pos_data.time_k[-1]
        t_min_k = t_now_k - WINDOW_SECONDS
        ind_k = [i for i, t in enumerate(pos_data.time_k) if t >= t_min_k]

        # Raw adatok szűrése
        if not pos_data.time_r:
            return
        t_now_r = pos_data.time_r[-1]
        t_min_r = t_now_r - WINDOW_SECONDS
        ind_r = [i for i, t in enumerate(pos_data.time_r) if t >= t_min_r]

        # Az indexek alapján szeletelünk
        xk = [pos_data.x_k[i] for i in ind_k]
        yk = [pos_data.y_k[i] for i in ind_k]
        zk = [pos_data.z_k[i] for i in ind_k]

        xr = [pos_data.x_r[i] for i in ind_r]
        yr = [pos_data.y_r[i] for i in ind_r]
        zr = [pos_data.z_r[i] for i in ind_r]

    # Frissítjük a vonalakat
    line_kalman.set_data(xk, yk)
    line_kalman.set_3d_properties(zk)

    #line_raw.set_data(xr, yr)
    #line_raw.set_3d_properties(zr)

ani = animation.FuncAnimation(fig, animate, interval=100)

plt.tight_layout()
plt.show()
