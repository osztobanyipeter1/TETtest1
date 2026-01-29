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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointCloudViewer:
    def __init__(self, point_cloud_file, host='127.0.0.1', port=12345, scale_factor=0.5):
        # Socket szerver inicializálása
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192) 
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print(f"Socket szerver indítva: {host}:{port}, várjuk a kapcsolatot...")
        
        # Kapcsolat elfogadása
        self.client_socket, self.client_address = self.server_socket.accept()
        print(f"Kapcsolat elfogadva: {self.client_address}")
        
        self.lock = threading.Lock()
        
        self.running = True
        self.thread = threading.Thread(target=self.receive_data)
        self.thread.daemon = True
        self.thread.start()

        # Pontfelhő betöltése és MÉRETEZÉSE
        print("Pontfelhő betöltése és méretezése...")
        self.pcd = o3d.io.read_point_cloud(point_cloud_file)
        
        # Pontfelhő méretezése
        self.scale_point_cloud(scale_factor)
        
        self.vertices = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else np.ones_like(self.vertices) * 0.7

        # STATIKUS MESH GENERÁLÁS - CSAK EGYSZER, ELŐRE
        print("Statikus mesh generálása...")
        self.mesh_vertices, self.mesh_triangles = self.generate_static_mesh()
        if self.mesh_vertices is not None:
            print(f"Mesh generálva: {len(self.mesh_vertices)} csúcs, {len(self.mesh_triangles)} háromszög")
        else:
            print("Nem sikerült mesh-t generálni")

        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)
        print(f"Pontfelhő középpontja: {self.center}")
        print(f"Kiterjedése: min={self.bounds_min}, max={self.bounds_max}")

        self.camera_pos = self.center + np.array([0.0, 0.0, 2.0], dtype=np.float32)  # Közelebb kezd
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.max_distance = 3.0  # Kisebb távolság
        self.fov_cos = np.cos(np.radians(90))
        self.point_size = 3.0
        self.movement_speed = 0.5  # Lassabb mozgás

        # Mesh megjelenítési beállítások
        self.show_mesh = True
        self.mesh_alpha = 0.1
        self.mesh_color = [0.2, 0.6, 1.0]  # Kék szín
        self.show_wireframe = True
        self.wireframe_color = [0.0, 0.0, 0.0]  # Fekete drótváz

        pygame.init()
        self.display = (1600, 900)  # Kisebb ablak
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("PointCloud Viewer - Fő ablak")
        pygame.mouse.set_visible(True)

        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)  # Kisebb távolság
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glPointSize(self.point_size)

        self.quaternion_w = 1.0
        self.quaternion_x = 0.0
        self.quaternion_y = 0.0
        self.quaternion_z = 0.0

        # Koordináta-rendszer ablak inicializálása
        self.setup_coordinate_window()

    def scale_point_cloud(self, scale_factor):
        """Pontfelhő méretezése kisebbre"""
        print(f"Pontfelhő méretezése {scale_factor} arányban...")
        
        # Pontok méretezése
        points = np.asarray(self.pcd.points)
        scaled_points = points * scale_factor
        self.pcd.points = o3d.utility.Vector3dVector(scaled_points)
        
        # Statisztikák kiírása
        original_bounds = np.ptp(points, axis=0)  # Original size
        scaled_bounds = np.ptp(scaled_points, axis=0)  # Scaled size
        print(f"Eredeti méret: {original_bounds}")
        print(f"Méretezett méret: {scaled_bounds}")
        print(f"Méretezési arány: {scale_factor}")

    def setup_coordinate_window(self):
        """Koordináta-rendszer ablak beállítása"""
        plt.ion()  # Interaktív mód
        self.coord_fig = plt.figure(figsize=(8, 6))
        self.coord_ax = self.coord_fig.add_subplot(111, projection='3d')
        self.coord_fig.canvas.manager.set_window_title('Koordináta Rendszer - Kamera Pozíció')
        
        # Kezdeti beállítások
        self.coord_ax.set_xlabel('X')
        self.coord_ax.set_ylabel('Y')
        self.coord_ax.set_zlabel('Z')
        self.coord_ax.set_title('Kamera Pozíció és Orientáció')
        
        # Pontfelhő határai
        margin = 1.0
        self.coord_ax.set_xlim([self.bounds_min[0] - margin, self.bounds_max[0] + margin])
        self.coord_ax.set_ylim([self.bounds_min[1] - margin, self.bounds_max[1] + margin])
        self.coord_ax.set_zlim([self.bounds_min[2] - margin, self.bounds_max[2] + margin])
        
        # Pontfelhő megjelenítése (csak pontokként)
        self.coord_ax.scatter(self.vertices[::10, 0], self.vertices[::10, 1], self.vertices[::10, 2], 
                             c='lightgray', s=1, alpha=0.3, label='Pontfelhő')
        
        # Kezdeti kamera pozíció
        self.camera_scatter = self.coord_ax.scatter([], [], [], c='red', s=100, label='Kamera')
        self.camera_quiver = None
        
        self.coord_ax.legend()
        plt.tight_layout()
        plt.draw()

    def update_coordinate_window(self):
        """Koordináta-rendszer ablak frissítése"""
        if not hasattr(self, 'coord_ax'):
            return
            
        self.coord_ax.clear()
        
        # Pontfelhő megjelenítése
        self.coord_ax.scatter(self.vertices[::10, 0], self.vertices[::10, 1], self.vertices[::10, 2], 
                             c='lightgray', s=1, alpha=0.3, label='Pontfelhő')
        
        # Kamera pozíció
        self.coord_ax.scatter([self.camera_pos[0]], [self.camera_pos[1]], [self.camera_pos[2]], 
                             c='red', s=100, label='Kamera')
        
        # Kamera orientáció (nyíl)
        arrow_length = 0.5
        arrow_end = self.camera_pos + self.camera_front * arrow_length
        
        self.coord_ax.quiver(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
                            self.camera_front[0] * arrow_length, 
                            self.camera_front[1] * arrow_length, 
                            self.camera_front[2] * arrow_length,
                            color='blue', arrow_length_ratio=0.2, linewidth=2, label='Nézet iránya')
        
        # Tengelyek
        axis_length = 1.0
        self.coord_ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X')
        self.coord_ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y')
        self.coord_ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z')
        
        # Beállítások
        margin = 1.0
        self.coord_ax.set_xlim([self.bounds_min[0] - margin, self.bounds_max[0] + margin])
        self.coord_ax.set_ylim([self.bounds_min[1] - margin, self.bounds_max[1] + margin])
        self.coord_ax.set_zlim([self.bounds_min[2] - margin, self.bounds_max[2] + margin])
        
        self.coord_ax.set_xlabel('X')
        self.coord_ax.set_ylabel('Y')
        self.coord_ax.set_zlabel('Z')
        
        # Pozíció információ
        pos_info = f'Pozíció: ({self.camera_pos[0]:.2f}, {self.camera_pos[1]:.2f}, {self.camera_pos[2]:.2f})'
        self.coord_ax.set_title(f'Kamera Pozíció és Orientáció\n{pos_info}')
        
        self.coord_ax.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def generate_static_mesh(self, alpha=0.6):
        """Statikus mesh generálása a teljes pontfelhőből - csak egyszer fut le"""
        try:
            print("Alpha shape mesh generálása...")
            
            # Pontfelhő leszűrése a jobb teljesítmény érdekében
            if len(self.pcd.points) > 100000:
                print("Pontfelhő leszűrése a mesh generáláshoz...")
                downsampled_pcd = self.pcd.voxel_down_sample(voxel_size=0.05)
            else:
                downsampled_pcd = self.pcd
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(downsampled_pcd, alpha)
            
            if len(mesh.vertices) == 0:
                print("Nem sikerült mesh-t generálni, próbálok kisebb alpha értékkel...")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(downsampled_pcd, alpha/2)
            
            # Mesh optimalizálása
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()
            
            # Normál vektorok számítása a megfelelő árnyékoláshoz
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            print(f"Statikus mesh generálva: {len(vertices)} csúcs, {len(triangles)} háromszög")
            return vertices, triangles
            
        except Exception as e:
            print(f"Hiba a statikus mesh generálásakor: {e}")
            return None, None

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
                            
                            # Orientáció frissítése
                            orientation = pose_data['orientation']
                            with self.lock:
                                self.quaternion_w = orientation['w']
                                self.quaternion_x = orientation['x'] 
                                self.quaternion_y = orientation['y']
                                self.quaternion_z = orientation['z']
                                
                            #print(f"Frissített pozíció: [{self.camera_pos[0]:.2f}, {self.camera_pos[1]:.2f}, {self.camera_pos[2]:.2f}], "
                            #      f"Orientáció: w={orientation['w']:.2f}, x={orientation['x']:.2f}, y={orientation['y']:.2f}, z={orientation['z']:.2f}")
                            
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
                elif event.key == pygame.K_m:
                    # Mesh láthatóság váltása
                    self.show_mesh = not self.show_mesh
                    print(f"Mesh megjelenítése: {'BE' if self.show_mesh else 'KI'}")
                elif event.key == pygame.K_w:
                    # Drótváz láthatóság váltása
                    self.show_wireframe = not self.show_wireframe
                    print(f"Drótváz megjelenítése: {'BE' if self.show_wireframe else 'KI'}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Mesh átlátszóság növelése
                    self.mesh_alpha = min(1.0, self.mesh_alpha + 0.1)
                    print(f"Mesh átlátszóság: {self.mesh_alpha:.1f}")
                elif event.key == pygame.K_MINUS:
                    # Mesh átlátszóság csökkentése
                    self.mesh_alpha = max(0.1, self.mesh_alpha - 0.1)
                    print(f"Mesh átlátszóság: {self.mesh_alpha:.1f}")

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
            print(f"Max távolság: {self.max_distance:.1f}")
        if keys[pygame.K_DOWN]:
            self.max_distance = max(0.1, self.max_distance - 0.1)
            print(f"Max távolság: {self.max_distance:.1f}")

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

    def render_mesh(self):
        """Statikus mesh renderelése"""
        if not self.show_mesh or self.mesh_vertices is None or self.mesh_triangles is None:
            return

        # Átlátszó felület
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Felület renderelése
        glBegin(GL_TRIANGLES)
        glColor4f(self.mesh_color[0], self.mesh_color[1], self.mesh_color[2], self.mesh_alpha)
        for tri in self.mesh_triangles:
            for idx in tri:
                if idx < len(self.mesh_vertices):  # Biztonsági ellenőrzés
                    glVertex3fv(self.mesh_vertices[idx])
        glEnd()
        
        # Drótváz renderelése
        if self.show_wireframe:
            glDisable(GL_BLEND)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(1.0)
            glColor3f(self.wireframe_color[0], self.wireframe_color[1], self.wireframe_color[2])
            
            glBegin(GL_TRIANGLES)
            for tri in self.mesh_triangles:
                for idx in tri:
                    if idx < len(self.mesh_vertices):
                        glVertex3fv(self.mesh_vertices[idx])
            glEnd()
            
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_BLEND)
        
        glDisable(GL_BLEND)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.update_camera_orientation()

        cam_target = self.camera_pos + self.camera_front
        gluLookAt(*self.camera_pos, *cam_target, *self.camera_up)

        # Pontok renderelése
        visible_points, colors = self.get_visible_points()

        glBegin(GL_POINTS)
        for i, point in enumerate(visible_points):
            distance = np.linalg.norm(point - self.camera_pos)
            t = min(distance / self.max_distance, 1.0)

            if self.pcd.has_colors() and len(colors) > 0 and i < len(colors):
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

        # STATIKUS MESH RENDERELÉSE - mindig ugyanaz a mesh jelenik meg
        self.render_mesh()

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        last_print_time = time.time()
        last_coord_update = time.time()
        last_fps_time = time.time()
        frame_count = 0
        fps = 0
        running = True

        print("""
    Vezérlés:
    - W, A, S, D: Mozgás
    - SPACE, LSHIFT: Fel / Le
    - Fel/Le nyilak: Látható távolság változtatása
    - M: Mesh láthatóság váltása
    - W: Drótváz láthatóság váltása
    - +/-: Mesh átlátszóság változtatása
    - ESC: Kilépés
        """)

        while running:
            delta_time = clock.tick(60) / 1000.0
            frame_count +=1
            running = self.process_input(delta_time)
            self.render()


            # Koordináta ablak frissítése (ritkábban a teljesítmény érdekében)
            current_time = time.time()

            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time- last_fps_time)
                frame_count = 0
                last_fps_time = current_time

            if current_time - last_coord_update > 0.05:  # 10 FPS
                self.update_coordinate_window()
                last_coord_update = current_time

            if current_time - last_print_time > 0.5:
                visible_points, _ = self.get_visible_points()
                print(f"\rLátható pontok: {len(visible_points):5d} | FPS: {fps:5.1f} | Pozíció: [{self.camera_pos[0]:6.2f}, {self.camera_pos[1]:6.2f}, {self.camera_pos[2]:6.2f}]", end="", flush=True)

        self.running = False
        self.thread.join()
        self.client_socket.close()
        self.server_socket.close()
        plt.close('all')
        pygame.quit()

if __name__ == "__main__":
    # Scale factor: 0.5 = felére kicsinyítés, 0.25 = negyedére, stb.
    viewer = PointCloudViewer("centered_sampled20000.ply", host='127.0.0.1', scale_factor=0.1)
    viewer.run()