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
from threading import Thread, Event
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict



class PointCloudViewer:
    def __init__(self, point_cloud_file, host='127.0.0.1', port=12345, scale_factor=0.5):
        # ============ SOCKET INICIALIZÁLÁS TIMEOUT-TAL ============
        self.socket_connected = False
        self.socket_timeout_event = Event()
        
        # Socket szerver inicializálása és timeout-os várakozás
        self.setup_socket_with_timeout(host, port, timeout=5.0)
        
        # Ha nincs TCP kapcsolat, fallback módra váltunk
        if not self.socket_connected:
            print("\n" + "="*60)
            print("TCP KAPCSOLAT SIKERTELEN - FALLBACK MÓD AKTIVÁLVA")
            print("="*60)
            print("Vezérlés: Billentyűzet (W/A/S/D) + Egér (kamera forgatás)")
            print("Egér lock/unlock: P billentyű")
            print("="*60 + "\n")
            
            self.client_socket = None
            self.thread = None
            
            # Mouse state inicializálása fallback módhoz (FPS-szerű)
            self.mouse_locked = False
            self.mouse_sensitivity = 0.1
            self.pitch = 0.0   # Függőleges szög (fok)
            self.yaw = -90.0   # Vízszintes szög (fok)
        else:
            print("\n✓ TCP kapcsolat sikeresen felépítve")
            # Threading csak ha van TCP kapcsolat
            self.lock = threading.Lock()
            self.thread = threading.Thread(target=self.receive_data)
            self.thread.daemon = True
            self.thread.start()
        
        self.running = True


        # ============ PONTFELHŐ BETÖLTÉSE ============
        print("Pontfelhő betöltése és méretezése...")
        self.pcd = o3d.io.read_point_cloud(point_cloud_file)
        
        self.scale_point_cloud(scale_factor)
        
        self.vertices = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else np.ones_like(self.vertices) * 0.7


        self.mesh_alpha = 0.6


        # ============ KAMERA BEÁLLÍTÁSOK ============
        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)
        print(f"Pontfelhő középpontja: {self.center}")
        print(f"Kiterjedése: min={self.bounds_min}, max={self.bounds_max}\n")


        self.camera_pos = self.center + np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.max_distance = 1.8
        self.fov_cos = np.cos(np.radians(60))
        self.point_size = 3.0
        self.movement_speed = 1.0


        # ============ MESH MEGJELENÍTÉS ============
        self.show_mesh = True
        self.mesh_color = [0.2, 0.6, 1.0]
        self.show_wireframe = True
        self.wireframe_color = [0.0, 0.0, 0.0]
        self.current_mesh_vertices = None
        self.current_mesh_triangles = None
        self.last_visible_points_hash = None  # Hash-alapú cache


        # ============ PYGAME INICIALIZÁLÁSA ============
        pygame.init()
        self.display = (1500, 720)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("PointCloud Viewer - Fallback Mode Support")
        pygame.mouse.set_visible(True)


        # ============ OPENGL BEÁLLÍTÁSOK ============
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glPointSize(self.point_size)



        # ============ QUATERNION (TCP módhoz) ============
        if self.socket_connected:
            self.quaternion_w = 1.0
            self.quaternion_x = 0.0
            self.quaternion_y = 0.0
            self.quaternion_z = 0.0


        # ============ KOORDINÁTA ABLAK ============
        self.setup_coordinate_window()



    def setup_socket_with_timeout(self, host, port, timeout=5.0):
        """
        Socket szerver inicializálása timeout-tal.
        Ha 5s alatt nem jön kapcsolat, fallback módra vált.
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
            self.server_socket.bind((host, port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(timeout)  # TIMEOUT BEÁLLÍTÁSA
            
            print(f"Socket szerver indítva: {host}:{port}")
            print(f"Várakozás kapcsolatra (timeout: {timeout}s)...")
            
            try:
                # Timeout-os blokkoló accept
                self.client_socket, self.client_address = self.server_socket.accept()
                print(f"✓ Kapcsolat elfogadva: {self.client_address}")
                self.socket_connected = True
                
                # Lock socket kommunikációhoz
                self.lock = threading.Lock()
                
            except socket.timeout:
                print(f"✗ Socket timeout után {timeout}s - fallback módra váltás")
                self.socket_connected = False
                self.server_socket.close()
                
        except Exception as e:
            print(f"✗ Socket inicializálási hiba: {e} - fallback módra váltás")
            self.socket_connected = False



    def scale_point_cloud(self, scale_factor):
        """Pontfelhő méretezése kisebbre"""
        print(f"Pontfelhő méretezése {scale_factor} arányban...")
        
        points = np.asarray(self.pcd.points)
        scaled_points = points * scale_factor
        self.pcd.points = o3d.utility.Vector3dVector(scaled_points)
        
        original_bounds = np.ptp(points, axis=0)
        scaled_bounds = np.ptp(scaled_points, axis=0)
        print(f"Eredeti méret: {original_bounds}")
        print(f"Méretezett méret: {scaled_bounds}")



    def setup_coordinate_window(self):
        """Koordináta-rendszer ablak beállítása"""
        plt.ion()
        self.coord_fig = plt.figure(figsize=(8, 6))
        self.coord_ax = self.coord_fig.add_subplot(111, projection='3d')
        self.coord_fig.canvas.manager.set_window_title('Koordináta Rendszer - Kamera Pozíció')
        
        self.coord_ax.set_xlabel('X')
        self.coord_ax.set_ylabel('Y')
        self.coord_ax.set_zlabel('Z')
        self.coord_ax.set_title('Kamera Pozíció és Orientáció')
        
        margin = 1.0
        self.coord_ax.set_xlim([self.bounds_min[0] - margin, self.bounds_max[0] + margin])
        self.coord_ax.set_ylim([self.bounds_min[1] - margin, self.bounds_max[1] + margin])
        self.coord_ax.set_zlim([self.bounds_min[2] - margin, self.bounds_max[2] + margin])
        
        self.coord_ax.scatter(self.vertices[::10, 0], self.vertices[::10, 1], self.vertices[::10, 2], 
                             c='lightgray', s=1, alpha=0.3, label='Pontfelhő')
        
        self.camera_scatter = self.coord_ax.scatter([], [], [], c='red', s=100, label='Kamera')
        
        self.coord_ax.legend()
        plt.tight_layout()
        plt.draw()



    def update_coordinate_window(self):
        """Koordináta-rendszer ablak frissítése"""
        if not hasattr(self, 'coord_ax'):
            return
            
        self.coord_ax.clear()
        
        self.coord_ax.scatter(self.vertices[::10, 0], self.vertices[::10, 1], self.vertices[::10, 2], 
                             c='lightgray', s=1, alpha=0.3, label='Pontfelhő')
        
        self.coord_ax.scatter([self.camera_pos[0]], [self.camera_pos[1]], [self.camera_pos[2]], 
                             c='red', s=100, label='Kamera')
        
        arrow_length = 0.5
        self.coord_ax.quiver(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
                            self.camera_front[0] * arrow_length, 
                            self.camera_front[1] * arrow_length, 
                            self.camera_front[2] * arrow_length,
                            color='blue', arrow_length_ratio=0.2, linewidth=2, label='Nézet iránya')
        
        axis_length = 1.0
        self.coord_ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X')
        self.coord_ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y')
        self.coord_ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z')
        
        margin = 1.0
        self.coord_ax.set_xlim([self.bounds_min[0] - margin, self.bounds_max[0] + margin])
        self.coord_ax.set_ylim([self.bounds_min[1] - margin, self.bounds_max[1] + margin])
        self.coord_ax.set_zlim([self.bounds_min[2] - margin, self.bounds_max[2] + margin])
        
        self.coord_ax.set_xlabel('X')
        self.coord_ax.set_ylabel('Y')
        self.coord_ax.set_zlabel('Z')
        
        pos_info = f'Pozíció: ({self.camera_pos[0]:.2f}, {self.camera_pos[1]:.2f}, {self.camera_pos[2]:.2f})'
        mode_info = f"Mód: {'TCP' if self.socket_connected else 'FALLBACK'}"
        self.coord_ax.set_title(f'Kamera Pozíció\n{pos_info} | {mode_info}')
        
        self.coord_ax.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)



    def generate_mesh_from_visible_points(self, visible_points):
        """On-demand mesh generálása a látható pontokból (hash-alapú cache-vel)"""
        if len(visible_points) < 10:  # Minimum 10 pont
            return None, None
        
        try:
            # Hash-alapú cache ellenőrzése
            current_hash = hash(visible_points.tobytes())
            if current_hash == self.last_visible_points_hash:
                # Ugyanazok a pontok, mesh nem változott
                return self.current_mesh_vertices, self.current_mesh_triangles
            
            visible_pcd = o3d.geometry.PointCloud()
            visible_pcd.points = o3d.utility.Vector3dVector(visible_points)
            
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    visible_pcd, 
                    alpha=self.mesh_alpha
                )
            except:
                # Fallback: kisebb alpha érték
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    visible_pcd, 
                    alpha=self.mesh_alpha / 2
                )
            
            if len(mesh.vertices) == 0:
                return None, None
            
            # Mesh optimalizálása
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()
            
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Cache frissítése
            self.last_visible_points_hash = current_hash
            
            return vertices, triangles
            
        except Exception as e:
            return None, None



    def receive_data(self):
        """TCP socketből adatok fogadása (csak TCP módban)"""
        if not self.socket_connected or self.client_socket is None:
            return
            
        buffer = ""
        while self.running and self.socket_connected:
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
                            
                            position = pose_data['position']
                            self.camera_pos = np.array([
                                position['x'], 
                                position['y'], 
                                position['z']
                            ], dtype=np.float32)
                            
                            orientation = pose_data['orientation']
                            if self.socket_connected:
                                with self.lock:
                                    self.quaternion_w = orientation['w']
                                    self.quaternion_x = orientation['x'] 
                                    self.quaternion_y = orientation['y']
                                    self.quaternion_z = orientation['z']
                            
                        except (json.JSONDecodeError, KeyError) as e:
                            pass
            except Exception as e:
                print(f"Socket hiba: {e}")
                break



    def update_camera_orientation(self):
        """Kamera orientáció frissítése (TCP módban)"""
        if not self.socket_connected:
            return
            
        with self.lock:
            q_w = self.quaternion_w
            q_x = self.quaternion_x
            q_y = self.quaternion_y
            q_z = self.quaternion_z
            
            norm = np.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)
            if norm > 0:
                q_w /= norm
                q_x /= norm
                q_y /= norm
                q_z /= norm
            
            def quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z):
                R = np.array([
                    [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
                    [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
                    [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
                ])
                return R


            R = quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)
            self.camera_front = R[:, 2]
            self.camera_up = R[:, 1]
            
            self.camera_front /= np.linalg.norm(self.camera_front)
            self.camera_up /= np.linalg.norm(self.camera_up)



    def update_camera_orientation_fallback(self):
        """Kamera orientáció frissítése fallback módban (FPS-szerű egér kezelés)"""
        if self.socket_connected:
            return
        
        if not self.mouse_locked:
            return
        
        # Egér delta mozgás (relatív)
        dx, dy = pygame.mouse.get_rel()
        
        if dx != 0 or dy != 0:
            # Yaw és pitch frissítése (fokokban)
            self.yaw += dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            
            # Pitch korlátozása (-89 és 89 fok között)
            self.pitch = max(-89.0, min(89.0, self.pitch))
        
        # Kamera front vektor számítása (fokból radiánba konvertálva)
        self.camera_front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ], dtype=np.float32)
        
        self.camera_front /= np.linalg.norm(self.camera_front)
        
        # Up vektor számítása
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(self.camera_front, world_up)
        right /= np.linalg.norm(right)
        self.camera_up = np.cross(right, self.camera_front)
        self.camera_up /= np.linalg.norm(self.camera_up)



    def process_input(self, delta_time):
        """Bemenet kezelése (billentyűzet + egér)"""
        running = True
        move_direction = np.zeros(3, dtype=np.float32)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_m:
                    self.show_mesh = not self.show_mesh
                    print(f"Mesh megjelenítése: {'BE' if self.show_mesh else 'KI'}")
                elif event.key == pygame.K_x:
                    self.show_wireframe = not self.show_wireframe
                    print(f"Drótváz megjelenítése: {'BE' if self.show_wireframe else 'KI'}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.mesh_alpha = min(1.0, self.mesh_alpha + 0.01)
                    print(f"Mesh alpha: {self.mesh_alpha:.2f}")
                    
                elif event.key == pygame.K_MINUS:
                    self.mesh_alpha = max(0.01, self.mesh_alpha - 0.01)
                    print(f"Mesh alpha: {self.mesh_alpha:.2f}")
                    
                elif event.key == pygame.K_p:
                    # Egér lock/unlock fallback módban
                    if not self.socket_connected:
                        self.mouse_locked = not self.mouse_locked
                        pygame.mouse.set_visible(not self.mouse_locked)
                        pygame.event.set_grab(self.mouse_locked)
                        print(f"Egér lock: {'BE' if self.mouse_locked else 'KI'}")


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


        if keys[pygame.K_UP]:
            self.max_distance += 0.1
        if keys[pygame.K_DOWN]:
            self.max_distance = max(0.1, self.max_distance - 0.1)


        if np.linalg.norm(move_direction) > 0:
            move_direction /= np.linalg.norm(move_direction)
            self.camera_pos += move_direction * self.movement_speed * delta_time


        return running



    def get_visible_points(self):
        """Látható pontok szűrése"""
        directions = self.vertices - self.camera_pos
        distances = np.linalg.norm(directions, axis=1)
        directions_normalized = directions / (distances[:, np.newaxis] + 1e-8)


        dot = np.dot(directions_normalized, self.camera_front)
        mask = (distances < self.max_distance) & (dot > self.fov_cos)


        return self.vertices[mask], self.colors[mask]



    def render_mesh(self):
        """Dinamikus szín mesh renderelése"""
        if not self.show_mesh or self.current_mesh_vertices is None or self.current_mesh_triangles is None:
            return

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Felület renderelése dinamikus szín gradienssel
        glBegin(GL_TRIANGLES)
        for tri in self.current_mesh_triangles:
            for idx in tri:
                if idx < len(self.current_mesh_vertices):
                    vertex = self.current_mesh_vertices[idx]
                    
                    # Távolság alapú szín számítás
                    distance = np.linalg.norm(vertex - self.camera_pos)
                    t = min(distance / self.max_distance, 1.0)
                    
                    # Szín interpoláció (cián -> magenta)
                    if t < 0.5:
                        t2 = t * 2
                        r = t2
                        g = 1.0 - t2
                        b = 1.0
                    else:
                        t2 = (t - 0.5) * 2
                        r = 1.0
                        g = 1.0 - t2
                        b = 1.0 - t2
                    
                    glColor4f(r, g, b, 1.0)
                    glVertex3fv(vertex)
        glEnd()
        
        # Drótváz renderelése (fekete vonalak)
        if self.show_wireframe:
            glLineWidth(2.0)
            glColor4f(0.0, 0.0, 0.0, 1.0)
            for tri in self.current_mesh_triangles:
                glBegin(GL_LINE_LOOP)
                for idx in tri:
                    if idx < len(self.current_mesh_vertices):
                        glVertex3fv(self.current_mesh_vertices[idx])
                glEnd()
        
        glDisable(GL_BLEND)



    def render(self):
        """Renderelési loop"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


        # Orientáció frissítése (TCP vagy fallback)
        if self.socket_connected:
            self.update_camera_orientation()
        else:
            self.update_camera_orientation_fallback()


        cam_target = self.camera_pos + self.camera_front
        gluLookAt(*self.camera_pos, *cam_target, *self.camera_up)


        visible_points, colors = self.get_visible_points()


        if self.show_mesh and len(visible_points) > 0:
            mesh_vertices, mesh_triangles = self.generate_mesh_from_visible_points(visible_points)
            self.current_mesh_vertices = mesh_vertices
            self.current_mesh_triangles = mesh_triangles


        # Pontok kirajzolása (dinamikus szín)
        glBegin(GL_POINTS)
        for point in visible_points:
            distance = np.linalg.norm(point - self.camera_pos)
            t = min(distance / self.max_distance, 1.0)

            if t < 0.5:
                t2 = t * 2
                r = t2
                g = 1.0 - t2
                b = 1.0
            else:
                t2 = (t - 0.5) * 2
                r = 1.0
                g = 1.0 - t2
                b = 1.0 - t2

            glColor3f(r, g, b)
            glVertex3fv(point)
        glEnd()


        self.render_mesh()


        pygame.display.flip()



    def run(self):
        """Főciklus"""
        clock = pygame.time.Clock()
        last_print_time = time.time()
        last_coord_update = time.time()
        last_fps_time = time.time()
        frame_count = 0
        fps = 0
        running = True


        mode_str = "TCP Mód" if self.socket_connected else "FALLBACK Mód (Egér + Billentyűzet)"
        print(f"\n{'='*60}")
        print(f"Indítva: {mode_str}")
        print(f"{'='*60}")
        print("""
    Vezérlés:
    - W, A, S, D: Mozgás előre/bal/hátra/jobb
    - SPACE, LSHIFT: Fel / Le
    - Fel/Le nyilak: Látható távolság változtatása
    - M: Mesh láthatóság váltása
    - W: Drótváz láthatóság váltása
    - +/-: Mesh alpha paraméter
        """)
        
        if not self.socket_connected:
            print("    FALLBACK MÓD SPECIFIKUS:")
            print("    - Egér mozgatás: Kamera forgatása (pitch/yaw)")
            print("    - P: Egér lock/unlock")
            print("")


        while running:
            delta_time = clock.tick(60) / 1000.0
            frame_count += 1
            running = self.process_input(delta_time)
            self.render()


            current_time = time.time()


            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time


            if current_time - last_coord_update > 0.05:
                self.update_coordinate_window()
                last_coord_update = current_time


            if current_time - last_print_time > 0.5:
                visible_points, _ = self.get_visible_points()
                mode_char = "TCP" if self.socket_connected else "FB"
                print(f"\rLátható: {len(visible_points):5d} | FPS: {fps:.2f} | Alpha: {self.mesh_alpha:.2f} | Pos: [{self.camera_pos[0]:6.2f}, {self.camera_pos[1]:6.2f}, {self.camera_pos[2]:6.2f}]", end="", flush=True)


        self.running = False
        
        if self.socket_connected and self.thread:
            self.thread.join()
            self.client_socket.close()
            self.server_socket.close()
        
        plt.close('all')
        pygame.quit()



if __name__ == "__main__":
    viewer = PointCloudViewer("centered_sampled20000.ply", host='127.0.0.1', scale_factor=0.1)
    viewer.run()