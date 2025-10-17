import open3d as o3d
import time
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import socket
import json
import threading

class PointCloudViewer:
    def __init__(self, point_cloud_file, host='127.0.0.1', port=12345):
        # Socket szerver inicializálása
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print(f"Socket szerver indítva: {host}:{port}, várjuk a kapcsolatot...")
        
        # Kapcsolat elfogadása
        self.client_socket, self.client_address = self.server_socket.accept()
        print(f"Kapcsolat elfogadva: {self.client_address}")
        
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        self.filtered_yaw = 0.0
        self.lock = threading.Lock()
        
        self.running = True
        self.thread = threading.Thread(target=self.receive_data)
        self.thread.daemon = True
        self.thread.start()

        self.pcd = o3d.io.read_point_cloud(point_cloud_file)
        self.vertices = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else np.ones_like(self.vertices) * 0.7

        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)

        print(f"Pontfelhő középpontja: {self.center}")
        print(f"Kiterjedése: min={self.bounds_min}, max={self.bounds_max}")

        self.camera_pos = self.center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.max_distance = 5.0
        self.fov_cos = np.cos(np.radians(45))
        self.point_size = 3.0
        self.movement_speed = 1.0

        # Quaternion változók
        self.quaternion_w = 1.0
        self.quaternion_x = 0.0
        self.quaternion_y = 0.0
        self.quaternion_z = 0.0

        pygame.init()
        self.display = (1980, 1200)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.mouse.set_visible(True)

        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glPointSize(self.point_size)

    def receive_data(self):
        buffer = ""
        while self.running:
            try:
                data = self.client_socket.recv(1024).decode()
                if not data:
                    print("Nincs adat, a kapcsolat lezárult")
                    break
                    
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            pose_data = json.loads(line)
                            
                            # Pozíció frissítése
                            position = pose_data['position']
                            self.camera_pos = np.array([
                                position['x'], 
                                position['y'], 
                                position['z']
                            ], dtype=np.float32)
                            
                            # Orientáció frissítése (quaternion)
                            orientation = pose_data['orientation']
                            with self.lock:
                                self.quaternion_w = orientation['w']
                                self.quaternion_x = orientation['x'] 
                                self.quaternion_y = orientation['y']
                                self.quaternion_z = orientation['z']
                                
                            print(f"Frissített pozíció: [{self.camera_pos[0]:.2f}, {self.camera_pos[1]:.2f}, {self.camera_pos[2]:.2f}], "
                                  f"Orientáció: w={orientation['w']:.2f}, x={orientation['x']:.2f}, y={orientation['y']:.2f}, z={orientation['z']:.2f}")
                            
                        except json.JSONDecodeError as e:
                            print(f"Érvénytelen adat: {line}, Hiba: {e}")
                        except KeyError as e:
                            print(f"Hiányzó kulcs az adatokban: {e}, Sor: {line}")
            except Exception as e:
                print(f"Socket hiba: {e}")
                break

    def update_camera_orientation(self):
        with self.lock:
            q_w = self.quaternion_w
            q_x = self.quaternion_x
            q_y = self.quaternion_y
            q_z = self.quaternion_z
            
            # Normalizálás
            norm = np.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)
            if norm > 0:
                q_w /= norm
                q_x /= norm
                q_y /= norm
                q_z /= norm
            
            # Kvaternióból forgatási mátrix
            def quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z):
                R = np.array([
                    [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
                    [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
                    [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
                ])
                return R

            R = quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)
            self.camera_front = R[:, 2]  # Z tengely = előre
            self.camera_up = R[:, 1]     # Y tengely = felfelé
            
            # Normalizálás
            self.camera_front /= np.linalg.norm(self.camera_front)
            self.camera_up /= np.linalg.norm(self.camera_up)

    def process_input(self, delta_time):
        running = True
        move_direction = np.zeros(3, dtype=np.float32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            move_direction += self.camera_front
        if keys[pygame.K_s]:
            move_direction -= self.camera_front
        if keys[pygame.K_a]:
            right = np.cross(self.camera_front, self.camera_up)
            move_direction -= right / np.linalg.norm(right)
        if keys[pygame.K_d]:
            right = np.cross(self.camera_front, self.camera_up)
            move_direction += right / np.linalg.norm(right)
        if keys[pygame.K_SPACE]:
            move_direction += self.camera_up
        if keys[pygame.K_LSHIFT]:
            move_direction -= self.camera_up

        # További vezérlési lehetőségek
        if keys[pygame.K_UP]:
            self.max_distance += 0.1
            print(f"Max távolság: {self.max_distance:.1f}")
        if keys[pygame.K_DOWN]:
            self.max_distance = max(0.1, self.max_distance - 0.1)
            print(f"Max távolság: {self.max_distance:.1f}")
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            self.point_size += 0.5
            glPointSize(self.point_size)
            print(f"Pont méret: {self.point_size:.1f}")
        if keys[pygame.K_MINUS]:
            self.point_size = max(1.0, self.point_size - 0.5)
            glPointSize(self.point_size)
            print(f"Pont méret: {self.point_size:.1f}")

        if np.linalg.norm(move_direction) > 0:
            move_direction /= np.linalg.norm(move_direction)
            self.camera_pos += move_direction * self.movement_speed * delta_time

        return running

    def get_visible_points(self):
        directions = self.vertices - self.camera_pos
        distances = np.linalg.norm(directions, axis=1)
        directions_normalized = directions / distances[:, np.newaxis]

        dot = np.dot(directions_normalized, self.camera_front)
        mask = (distances < self.max_distance) & (dot > self.fov_cos)

        return self.vertices[mask], self.colors[mask]

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.update_camera_orientation()
        
        cam_target = self.camera_pos + self.camera_front
        gluLookAt(*self.camera_pos, *cam_target, *self.camera_up)

        visible_points, colors = self.get_visible_points()

        glBegin(GL_POINTS)
        for i, point in enumerate(visible_points):
            distance = np.linalg.norm(point - self.camera_pos)
            t = min(distance / self.max_distance, 1.0)

            # Szín számítása távolság alapján vagy eredeti színek használata
            if self.pcd.has_colors() and len(colors) > 0:
                # Eredeti színek használata
                glColor3f(colors[i][0], colors[i][1], colors[i][2])
            else:
                # Távolság alapú színátmenet
                if t < 0.5:
                    r = 1.0 - 2 * t
                    g = 2 * t
                    b = 0.0
                else:
                    t2 = (t - 0.5) * 2
                    r = 0.0
                    g = 1.0 - t2
                    b = t2
                glColor3f(r, g, b)
                
            glVertex3fv(point)
        glEnd()

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        last_print_time = time.time()
        running = True

        print("""
    Vezérlés:
    - W, A, S, D: Mozgás
    - SPACE, LSHIFT: Fel / Le
    - +/-: Pontok méretének változtatása
    - Fel/Le nyilak: Látható távolság változtatása
    - ESC: Kilépés
        """)

        while running:
            delta_time = clock.tick(60) / 1000.0
            running = self.process_input(delta_time)
            self.render()

            if time.time() - last_print_time > 2.0:  # Csak 2 másodpercenként írjon ki
                visible_points, _ = self.get_visible_points()
                with self.lock:
                    print(f"Látható pontok: {len(visible_points)}, Pozíció: [{self.camera_pos[0]:.2f}, {self.camera_pos[1]:.2f}, {self.camera_pos[2]:.2f}]")
                last_print_time = time.time()

        self.running = False
        self.thread.join()
        self.client_socket.close()
        self.server_socket.close()
        pygame.quit()

if __name__ == "__main__":
    viewer = PointCloudViewer("centered_sampled2.ply", host='127.0.0.1')
    viewer.run()