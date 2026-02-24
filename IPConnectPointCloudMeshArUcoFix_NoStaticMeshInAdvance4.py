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
from threading import Event
import matplotlib.pyplot as plt
import ctypes


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
            
            self.client_socket = None
            self.thread = None
            
            # Mouse state inicializálása fallback módhoz (FPS-szerű)
            self.mouse_locked = False
            self.mouse_sensitivity = 0.1
            self.pitch = 0.0   # Függőleges szög (fok)
            self.yaw = -90.0   # Vízszintes szög (fok)
        else:
            print("TCP kapcsolat sikeresen felépítve")
            # Threading csak ha van TCP kapcsolat
            self.lock = threading.Lock()
            self.thread = threading.Thread(target=self.receive_data)
            self.thread.daemon = True
            self.thread.start()
        
        self.running = True

        # ============ PONTFELHŐ BETÖLTÉSE CLUSTER ADATOKKAL ============
        self.load_point_cloud_with_clusters(point_cloud_file)
        
        self.scale_point_cloud(scale_factor)
        
        self.vertices = np.asarray(self.pcd.points)
        self.clusters = self.cluster_data  # Cluster értékek tárolása
        
        # Cluster szűrő beállítások
        self.cluster_filter_enabled = False
        self.selected_clusters = set()  # Kiválasztott cluster értékek
        self.cluster_display_mode = "normal"  # "normal", "single", "multi"
        
        # Színek generálása a cluster értékek alapján
        self.generate_cluster_colors()
        
        self.mesh_alpha = 0.6

        # ============ KAMERA BEÁLLÍTÁSOK ============
        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)

        self.camera_pos = self.center + np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.max_distance = 10
        self.fov_cos = np.cos(np.radians(60))
        self.point_size = 3.0
        self.movement_speed = 1.0
        self.lod_factor = 1  # 1=full, 2=half, 4=quarter
        self.coord_window_enabled = True

        self.frustum_culling_enabled = True
        self.lod_enabled = True
        self.mesh_caching_enabled = True
        self.point_size_lod_enabled = False
        self.backface_culling_enabled = False
        
        # ============ GPU/CPU RENDERELÉS MÓD ============
        self.use_gpu_rendering = True  # Toggle GPU/CPU között

        # ============ MESH MEGJELENÍTÉS ============
        self.show_mesh = True
        self.show_wireframe = True
        self.current_mesh_vertices = None
        self.current_mesh_triangles = None
        self.last_visible_points_hash = None  # Hash-alapú cache

        # ============ PYGAME INICIALIZÁLÁSA ============
        pygame.init()
        
        # ⭐ KOMPATIBILITÁSI PROFIL (OpenGL 2.1)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_COMPATIBILITY)
        
        self.display = (1500, 720)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("PointCloud Viewer - Cluster alapú színezés és szűrés")
        pygame.mouse.set_visible(True)

        # ============ OPENGL BEÁLLÍTÁSOK ============
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glPointSize(self.point_size)

        # ============ VBO INICIALIZÁLÁSA (OpenGL context után) ============
        self.vbo_initialized = False
        try:
            self.setup_vbo()
        except Exception as e:
            print(f"VBO inicializálási hiba: {e}")
            self.vbo_initialized = False

        # ============ QUATERNION (TCP módhoz) ============
        if self.socket_connected:
            self.quaternion_w = 1.0
            self.quaternion_x = 0.0
            self.quaternion_y = 0.0
            self.quaternion_z = 0.0

        # ============ KOORDINÁTA ABLAK ============
        self.setup_coordinate_window()
        
        # Cluster információk kiírása
        self.print_cluster_info()
        
    def print_cluster_info(self):
        """Cluster információk kiírása"""
        unique_clusters = np.unique(self.clusters)
        cluster_counts = {}
        for cluster in unique_clusters:
            cluster_counts[cluster] = np.sum(self.clusters == cluster)
        
        print(f"\n{'='*60}")
        print(f"CLUSTER STATISZTIKA")
        print(f"{'='*60}")
        print(f"Összes pont: {len(self.vertices)}")
        print(f"Egyedi clusterek száma: {len(unique_clusters)}")
        print(f"\nCluster eloszlás:")
        print(f"{'Cluster':<10} {'Darab':<10} {'Arány':<10}")
        print(f"{'-'*30}")
        
        for cluster in sorted(unique_clusters):
            count = cluster_counts[cluster]
            percentage = (count / len(self.vertices)) * 100
            print(f"{int(cluster):<10} {count:<10} {percentage:.2f}%")
        
        print(f"\nCluster értékek listája: {sorted(unique_clusters)}")
        print(f"{'='*60}\n")
        
        # Használati útmutató cluster szűréshez
        print("CLUSTER SZŰRÉS PARANCSOK:")
        print("  [1] - [9]: Cluster hozzáadása a szűrőhöz (0-9 közötti értékek)")
        print("  F6: Cluster szűrő be/ki")
        print("  F7: Összes cluster mutatása")
        print("  F8: Egyedi cluster mód (csak a kiválasztott)")
        print("  F9: Több cluster mód (kiválasztottak)")
        print("  F10: Szűrő törlése")
        print("  Jelenlegi kiválasztott clusterek: -\n")
    
    def load_point_cloud_with_clusters(self, filename):
        """PLY fájl betöltése cluster adatokkal (4 oszlop)"""
        try:
            # PLY fájl beolvasása szövegként
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Fejléc feldolgozása
            header_end = 0
            points_start = 0
            for i, line in enumerate(lines):
                if line.strip() == "end_header":
                    header_end = i
                    points_start = i + 1
                    break
            
            # Pontok beolvasása
            points = []
            clusters = []
            
            for line in lines[points_start:]:
                if line.strip():
                    values = line.strip().split()
                    if len(values) >= 4:
                        x, y, z, cluster = map(float, values[:4])
                        points.append([x, y, z])
                        clusters.append(int(cluster))  # cluster érték tárolása
            
            points = np.array(points, dtype=np.float64)
            clusters = np.array(clusters, dtype=np.int32)
            
            # Open3D pontfelhő létrehozása
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            
            # Cluster adatok tárolása
            self.cluster_data = clusters
            
            print(f"Betöltve: {len(points)} pont, {len(np.unique(clusters))} különböző cluster")
            
        except Exception as e:
            print(f"Hiba a PLY fájl betöltésekor: {e}")
            # Fallback: üres pontfelhő
            self.pcd = o3d.geometry.PointCloud()
            self.cluster_data = np.array([])
    
    def generate_cluster_colors(self):
        """Színek generálása a cluster értékek alapján"""
        if len(self.cluster_data) == 0:
            self.colors = np.ones_like(self.vertices) * 0.7
            return
        
        # Egyedi clusterek
        self.unique_clusters = np.unique(self.cluster_data)
        
        # Előre definiált színpaletta (könnyen megkülönböztethető színek)
        self.color_palette = [
            [0.0, 0.0, 1.0],  # kék (16-os cluster)
            [1.0, 1.0, 0.0],  # sárga (17-es cluster)
            [1.0, 0.0, 0.0],  # piros
            [0.0, 1.0, 0.0],  # zöld
            [1.0, 0.0, 1.0],  # magenta
            [0.0, 1.0, 1.0],  # cián
            [1.0, 0.5, 0.0],  # narancs
            [0.5, 0.0, 1.0],  # lila
            [0.5, 0.5, 0.5],  # szürke
            [1.0, 0.5, 0.5],  # világospiros
            [0.5, 1.0, 0.5],  # világoszöld
            [0.5, 0.5, 1.0],  # világoskék
        ]
        
        # Színek hozzárendelése clusterekhez
        self.cluster_to_color = {}
        for i, cluster_id in enumerate(sorted(self.unique_clusters)):
            color_idx = i % len(self.color_palette)
            self.cluster_to_color[cluster_id] = self.color_palette[color_idx]
            print(f"  Cluster {int(cluster_id)} -> RGB{self.color_palette[color_idx]}")
        
        # Színek generálása minden ponthoz
        self.colors = np.zeros((len(self.vertices), 3))
        for i, cluster in enumerate(self.cluster_data):
            self.colors[i] = self.cluster_to_color[cluster]
    
    def apply_cluster_filter(self, vertices, clusters, colors):
        """Cluster szűrő alkalmazása a pontokra"""
        if not self.cluster_filter_enabled or len(self.selected_clusters) == 0:
            return vertices, clusters, colors
        
        if self.cluster_display_mode == "single":
            # Csak egy cluster mutatása
            mask = np.isin(clusters, list(self.selected_clusters))
        else:  # "multi" mód
            # Több cluster mutatása
            mask = np.isin(clusters, list(self.selected_clusters))
        
        return vertices[mask], clusters[mask], colors[mask]
    
    def setup_vbo(self):
        """VBO (Vertex Buffer Object) inicializálása - GPU memória optimalizálás"""
        if not self.vertices.size > 0:
            raise ValueError("No vertices to upload to GPU")
        
        try:
            # Adatok float32 formátumra konvertálása
            vertices_data = self.vertices.astype(np.float32)
            colors_data = self.colors.astype(np.float32)
            
            print(f"  GPU upload: {len(vertices_data)} pont, "
                  f"{vertices_data.nbytes / 1024 / 1024:.2f} MB")
            
            # VAO (Vertex Array Object) létrehozása
            self.vao = glGenVertexArrays(1)
            glBindVertexArray(self.vao)
            
            # ========== POZÍCIÓKAT TARTALMAZÓ VBO ==========
            self.vbo_positions = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
            glBufferData(GL_ARRAY_BUFFER, vertices_data.nbytes, vertices_data, GL_STATIC_DRAW)
            
            # Pozíció attribútum pointer (0. attribútum)
            # 3 float per vertex, 12 byte stride, offset 0
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            
            # ========== SZÍN VBO ==========
            self.vbo_colors = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
            glBufferData(GL_ARRAY_BUFFER, colors_data.nbytes, colors_data, GL_STATIC_DRAW)
            
            # Szín attribútum pointer (1. attribútum)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            
            # ========== UNBIND ==========
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.vbo_initialized = True
            
        except Exception as e:
            self.vbo_initialized = False
            raise

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
                print(f"Kapcsolat elfogadva: {self.client_address}")
                self.socket_connected = True
                
                # Lock socket kommunikációhoz
                self.lock = threading.Lock()
                
            except socket.timeout:
                print(f"Socket timeout után {timeout}s - fallback módra váltás")
                self.socket_connected = False
                self.server_socket.close()
                
        except Exception as e:
            print(f"Socket inicializálási hiba: {e} - fallback módra váltás")
            self.socket_connected = False

    def scale_point_cloud(self, scale_factor):
        """Pontfelhő méretezése kisebbre"""
        points = np.asarray(self.pcd.points)
        scaled_points = points * scale_factor
        self.pcd.points = o3d.utility.Vector3dVector(scaled_points)
        self.vertices = scaled_points

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
        
        # Pontfelhő megjelenítése cluster színekkel a koordináta ablakban
        colors_display = self.colors[::10] if len(self.colors) > 10 else self.colors
        self.coord_ax.scatter(self.vertices[::10, 0], self.vertices[::10, 1], self.vertices[::10, 2], 
                             c=colors_display, s=1, alpha=0.3, label='Pontfelhő')
        
        self.coord_ax.legend()
        plt.tight_layout()
        plt.draw()

    def update_coordinate_window(self):
        """Koordináta-rendszer ablak frissítése"""
        if not hasattr(self, 'coord_ax'):
            return
            
        self.coord_ax.clear()
        
        # Cluster szűrő alkalmazása a koordináta ablakban is
        filtered_vertices, filtered_clusters, filtered_colors = self.apply_cluster_filter(
            self.vertices, self.clusters, self.colors
        )
        
        # Pontfelhő megjelenítése cluster színekkel
        step = max(1, len(filtered_vertices) // 1000)  # Max 1000 pont a teljesítményért
        colors_display = filtered_colors[::step] if len(filtered_colors) > step else filtered_colors
        self.coord_ax.scatter(filtered_vertices[::step, 0], filtered_vertices[::step, 1], filtered_vertices[::step, 2], 
                             c=colors_display, s=1, alpha=0.3, label='Pontfelhő')
        
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
        cluster_info = f"Clusterek: {len(np.unique(self.clusters))}"
        filter_info = f"Szűrő: {'BE' if self.cluster_filter_enabled else 'KI'} ({self.cluster_display_mode})"
        selected_info = f"Kiválasztva: {sorted(self.selected_clusters) if self.selected_clusters else '-'}"
        
        self.coord_ax.set_title(f'Kamera Pozíció\n{pos_info} | {mode_info} | {cluster_info}\n{filter_info} | {selected_info}')
        
        self.coord_ax.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def generate_mesh_from_visible_points(self, visible_points, visible_clusters):
        """On-demand mesh generálása a látható pontokból cluster színezéssel"""
        if len(visible_points) < 10:  # Minimum 10 pont
            return None, None, None
        
        try:
            # Hash-alapú cache ellenőrzése
            current_hash = hash(visible_points.tobytes())
            if current_hash == self.last_visible_points_hash and self.mesh_caching_enabled:
                # Ugyanazok a pontok, mesh nem változott
                return self.current_mesh_vertices, self.current_mesh_triangles, self.current_mesh_clusters
            
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
                return None, None, None
            
            # Mesh optimalizálása
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()
            
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Cluster értékek hozzárendelése a mesh vertexekhez (legközelebbi pont alapján)
            from scipy.spatial import KDTree
            tree = KDTree(visible_points)
            distances, indices = tree.query(vertices)
            mesh_clusters = visible_clusters[indices]
            
            # Cache frissítése
            self.last_visible_points_hash = current_hash
            self.current_mesh_clusters = mesh_clusters
            
            return vertices, triangles, mesh_clusters
            
        except Exception as e:
            return None, None, None

    def receive_data(self):
        """TCP socketből adatok fogadása (csak TCP módban)"""
        if not self.socket_connected or self.client_socket is None:
            return
            
        buffer = ""
        while self.running and self.socket_connected:
            try:
                data = self.client_socket.recv(1024).decode()
                if not data:
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
                elif event.key == pygame.K_3 or event.key == pygame.K_EQUALS:
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
                        
                elif event.key == pygame.K_g:
                    # GPU/CPU renderelés mód váltása
                    self.use_gpu_rendering = not self.use_gpu_rendering
                    render_mode = "GPU (VBO)" if self.use_gpu_rendering else "CPU (glBegin/glEnd)"
                    print(f"Renderelési mód: {render_mode}")
                    
                elif event.key == pygame.K_l:
                    # LOD módosítás: 1 → 2 → 4 → 1
                    lod_cycle = {1: 2, 2: 3, 3: 4, 4:5, 5:1}
                    self.lod_factor = lod_cycle[self.lod_factor]
                    print(f"LOD faktor: {self.lod_factor}x (minden {self.lod_factor}. pont renderelése)")
                    
                elif event.key == pygame.K_c:
                    # Koordináta ablak update be/ki (ablak nem zárul be)
                    self.coord_window_enabled = not self.coord_window_enabled
                    print(f"Koordináta ablak update: {'BE' if self.coord_window_enabled else 'KI'}")
                    
                elif event.key == pygame.K_F1:
                    # Frustum Culling Toggle
                    self.frustum_culling_enabled = not self.frustum_culling_enabled
                    print(f"Frustum Culling: {'BE' if self.frustum_culling_enabled else 'KI'}")
                    
                elif event.key == pygame.K_F2:
                    # LOD Toggle
                    self.lod_enabled = not self.lod_enabled
                    print(f"LOD: {'BE' if self.lod_enabled else 'KI'}")
                    
                elif event.key == pygame.K_F3:
                    # Mesh Caching Toggle
                    self.mesh_caching_enabled = not self.mesh_caching_enabled
                    print(f"Mesh Caching: {'BE' if self.mesh_caching_enabled else 'KI'}")
                    
                elif event.key == pygame.K_F4:
                    # Point Size LOD Toggle
                    self.point_size_lod_enabled = not self.point_size_lod_enabled
                    print(f"Point Size LOD: {'BE' if self.point_size_lod_enabled else 'KI'}")
                    
                elif event.key == pygame.K_F5:
                    # Back-Face Culling Toggle
                    self.backface_culling_enabled = not self.backface_culling_enabled
                    if self.backface_culling_enabled:
                        glEnable(GL_CULL_FACE)
                        glCullFace(GL_BACK)
                    else:
                        glDisable(GL_CULL_FACE)
                    print(f"Back-Face Culling: {'BE' if self.backface_culling_enabled else 'KI'}")
                    
                # ============ CLUSTER SZŰRÉS PARANCSOK ============
                elif event.key == pygame.K_F6:
                    # Cluster szűrő be/ki
                    self.cluster_filter_enabled = not self.cluster_filter_enabled
                    status = "BE" if self.cluster_filter_enabled else "KI"
                    print(f"Cluster szűrő: {status}")
                    
                elif event.key == pygame.K_F7:
                    # Összes cluster mutatása (szűrő kikapcsolása)
                    self.cluster_filter_enabled = False
                    self.selected_clusters = set()
                    self.cluster_display_mode = "normal"
                    print("Összes cluster megjelenítése")
                    
                elif event.key == pygame.K_F8:
                    # Egyedi cluster mód
                    if len(self.selected_clusters) > 0:
                        self.cluster_filter_enabled = True
                        self.cluster_display_mode = "single"
                        print(f"Egyedi cluster mód - kiválasztva: {sorted(self.selected_clusters)}")
                    else:
                        print("Nincs kiválasztva cluster! Először válassz ki egyet (1-9)")
                        
                elif event.key == pygame.K_F9:
                    # Több cluster mód
                    if len(self.selected_clusters) > 0:
                        self.cluster_filter_enabled = True
                        self.cluster_display_mode = "multi"
                        print(f"Több cluster mód - kiválasztva: {sorted(self.selected_clusters)}")
                    else:
                        print("Nincs kiválasztva cluster! Először válassz ki egyet (1-9)")
                        
                elif event.key == pygame.K_F10:
                    # Szűrő törlése
                    self.cluster_filter_enabled = False
                    self.selected_clusters = set()
                    self.cluster_display_mode = "normal"
                    print("Szűrő törölve - minden cluster megjelenik")
                    
                # Számbillentyűk 1-9 a cluster kiválasztásához
                elif event.key >= pygame.K_0 and event.key <= pygame.K_9:
                    # Szám felismerése
                    num = event.key - pygame.K_0
                    
                    # Megnézzük, hogy van-e ilyen cluster
                    if num in self.unique_clusters:
                        if num in self.selected_clusters:
                            self.selected_clusters.remove(num)
                            print(f"Cluster {num} eltávolítva a kiválasztásból")
                        else:
                            self.selected_clusters.add(num)
                            print(f"Cluster {num} hozzáadva a kiválasztáshoz")
                        
                        print(f"Jelenlegi kiválasztott clusterek: {sorted(self.selected_clusters)}")
                    else:
                        print(f"Nincs {num} értékű cluster az adatokban!")
                        
                # Kétjegyű számok kezelése (10 feletti cluster értékek)
                elif event.key == pygame.K_1 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Ctrl+1 pl. 10-es cluster
                    self.try_add_cluster(10)
                elif event.key == pygame.K_2 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.try_add_cluster(11)
                elif event.key == pygame.K_3 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.try_add_cluster(12)
                elif event.key == pygame.K_4 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.try_add_cluster(13)
                elif event.key == pygame.K_5 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.try_add_cluster(14)
                elif event.key == pygame.K_6 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.try_add_cluster(15)
                elif event.key == pygame.K_7 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.try_add_cluster(16)
                elif event.key == pygame.K_8 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.try_add_cluster(17)
                elif event.key == pygame.K_9 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.try_add_cluster(18)
                elif event.key == pygame.K_0 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.try_add_cluster(19)

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
    
    def try_add_cluster(self, cluster_value):
        """Segédfüggvény cluster érték hozzáadásához"""
        if cluster_value in self.unique_clusters:
            if cluster_value in self.selected_clusters:
                self.selected_clusters.remove(cluster_value)
                print(f"Cluster {cluster_value} eltávolítva a kiválasztásból")
            else:
                self.selected_clusters.add(cluster_value)
                print(f"Cluster {cluster_value} hozzáadva a kiválasztáshoz")
            print(f"Jelenlegi kiválasztott clusterek: {sorted(self.selected_clusters)}")
        else:
            print(f"Nincs {cluster_value} értékű cluster az adatokban!")

    def get_visible_points(self):
        """Látható pontok szűrése + LOD + Frustum Culling + Cluster szűrő"""
        
        # FRUSTUM CULLING
        if self.frustum_culling_enabled:
            directions = self.vertices - self.camera_pos
            distances = np.linalg.norm(directions, axis=1)
            directions_normalized = directions / (distances[:, np.newaxis] + 1e-8)
            
            dot = np.dot(directions_normalized, self.camera_front)
            mask = (distances < self.max_distance) & (dot > self.fov_cos)
        else:
            # Minden pont (csak távolság szűrés)
            distances = np.linalg.norm(self.vertices - self.camera_pos, axis=1)
            mask = distances < self.max_distance
        
        # LOD ALKALMAZÁSA
        visible_vertices = self.vertices[mask]
        visible_clusters = self.clusters[mask]
        visible_colors = self.colors[mask]
        
        if self.lod_enabled and self.lod_factor > 1:
            visible_vertices = visible_vertices[::self.lod_factor]
            visible_clusters = visible_clusters[::self.lod_factor]
            visible_colors = visible_colors[::self.lod_factor]
        
        # CLUSTER SZŰRŐ ALKALMAZÁSA
        visible_vertices, visible_clusters, visible_colors = self.apply_cluster_filter(
            visible_vertices, visible_clusters, visible_colors
        )
        
        return visible_vertices, visible_clusters, visible_colors

    def render_mesh(self):
        """Dinamikus szín mesh renderelése cluster színekkel"""
        if not self.show_mesh or self.current_mesh_vertices is None or self.current_mesh_triangles is None:
            return

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Felület renderelése cluster színekkel
        glBegin(GL_TRIANGLES)
        for tri in self.current_mesh_triangles:
            for idx in tri:
                if idx < len(self.current_mesh_vertices) and hasattr(self, 'current_mesh_clusters') and idx < len(self.current_mesh_clusters):
                    vertex = self.current_mesh_vertices[idx]
                    cluster = self.current_mesh_clusters[idx]
                    
                    # Cluster színének lekérése
                    if cluster in self.cluster_to_color:
                        color = self.cluster_to_color[cluster]
                    else:
                        color = [0.7, 0.7, 0.7]  # alapértelmezett szürke
                    
                    glColor4f(color[0], color[1], color[2], 1.0)
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

    def render_points_gpu(self, visible_points, visible_colors):
        """GPU renderelés VBO-val"""
        if not self.vbo_initialized or len(visible_points) == 0:
            return
        
        # VBO frissítés csak a látható pontokkal
        visible_vertices = visible_points.astype(np.float32)
        visible_colors = visible_colors.astype(np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        glBufferSubData(GL_ARRAY_BUFFER, 0, visible_vertices.nbytes, visible_vertices)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glBufferSubData(GL_ARRAY_BUFFER, 0, visible_colors.nbytes, visible_colors)
        
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, len(visible_vertices))
        glBindVertexArray(0)

    def render_points_cpu(self, visible_points, visible_colors):
        """CPU renderelés glBegin/glEnd-vel"""
        if len(visible_points) == 0:
            return
        
        # CPU renderelés
        glBegin(GL_POINTS)
        for i, point in enumerate(visible_points):
            # Point Size LOD - távolság alapú
            if self.point_size_lod_enabled:
                dist = np.linalg.norm(point - self.camera_pos)
                size = max(1.0, 3.0 * (1.0 - min(dist / self.max_distance, 1.0)))
                glPointSize(size)
            
            glColor3f(visible_colors[i, 0], visible_colors[i, 1], visible_colors[i, 2])
            glVertex3fv(point)
        glEnd()
        
        # Reset point size
        glPointSize(self.point_size)

    def render(self):
        """Renderelési loop"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Orientáció frissítése
        if self.socket_connected:
            self.update_camera_orientation()
        else:
            self.update_camera_orientation_fallback()

        cam_target = self.camera_pos + self.camera_front
        gluLookAt(*self.camera_pos, *cam_target, *self.camera_up)

        visible_points, visible_clusters, visible_colors = self.get_visible_points()

        # GPU vagy CPU renderelés
        if self.use_gpu_rendering:
            self.render_points_gpu(visible_points, visible_colors)
        else:
            self.render_points_cpu(visible_points, visible_colors)

        # Mesh renderelés
        if self.show_mesh and len(visible_points) > 0:
            mesh_vertices, mesh_triangles, mesh_clusters = self.generate_mesh_from_visible_points(visible_points, visible_clusters)
            self.current_mesh_vertices = mesh_vertices
            self.current_mesh_triangles = mesh_triangles
            self.current_mesh_clusters = mesh_clusters
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
    - P: Egér lock/unlock (fallback módban)
    - G: GPU/CPU renderelés mód váltása
    - M: Mesh láthatóság váltása
    - X: Drótváz láthatóság váltása
    - 3(+)/-: Mesh alpha paraméter
    - L: LOD faktor
    - C: Koordináta ablak update
    - F1: Frustum Culling
    - F2: LOD toggle
    - F3: Mesh Caching
    - F4: Point Size LOD
    - F5: Back-Face Culling
    
    CLUSTER SZŰRÉS:
    - 1-9: Cluster hozzáadása/eltávolítása (0-9 értékek)
    - Ctrl+1..Ctrl+0: 10-19 közötti cluster értékek
    - F6: Cluster szűrő be/ki
    - F7: Összes cluster mutatása
    - F8: Egyedi cluster mód (csak a kiválasztott)
    - F9: Több cluster mód (kiválasztottak)
    - F10: Szűrő törlése
        """)
        
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

            if current_time - last_coord_update > 0.05 and self.coord_window_enabled:
                self.update_coordinate_window()
                last_coord_update = current_time

            if current_time - last_print_time > 0.5:
                visible_points, visible_clusters, _ = self.get_visible_points()
                render_mode = "GPU" if self.use_gpu_rendering else "CPU"
                unique_visible_clusters = len(np.unique(visible_clusters)) if len(visible_clusters) > 0 else 0
                
                filter_status = "BE" if self.cluster_filter_enabled else "KI"
                selected = f"Kiválasztva: {sorted(self.selected_clusters) if self.selected_clusters else '-'}"
                
                print(f"\rLátható: {len(visible_points):5d} | Clusterek: {unique_visible_clusters:3d} | FPS: {fps:.2f} | Alpha: {self.mesh_alpha:.2f} | Render: {render_mode} | Szűrő: {filter_status} {selected}", end="", flush=True)
                last_print_time = current_time

        self.running = False
        
        if self.socket_connected and self.thread:
            self.thread.join()
            self.client_socket.close()
            self.server_socket.close()
        
        plt.close('all')
        pygame.quit()


if __name__ == "__main__":
    viewer = PointCloudViewer("pcexample.ply", host='127.0.0.1', scale_factor=0.1)
    viewer.run()