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
from scipy.spatial.transform import Rotation as R

class PointCloudViewer:
    def __init__(self, point_cloud_file, host='127.0.0.1', port=12345):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        print("Connected to AR glasses server")

        # Kvaternió + pozíció értékek
        self.quaternion = [0.0, 0.0, 0.0, 1.0] # [x, y, z, w]
        self.position = np.array([0.0, 0.0, 0.0])
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

        # Kamera kezdő értékei
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_pos = self.center + np.array([0.0, 0.0, 3.0], dtype=np.float32)

        self.max_distance = 5.0
        self.fov_cos = np.cos(np.radians(90))
        self.point_size = 3.0
        self.movement_speed = 1.0

        self.alpha_value = 0.6
        self.last_visible_hash = None
        self.mesh_triangles = None
        self.mesh_vertices = None

        pygame.init()
        self.display = (1280, 960)
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
                data = self.socket.recv(1024).decode()
                if not data:
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            data_json = json.loads(line)
                            with self.lock:
                                # [w, x, y, z, pos_x, pos_y, pos_z]
                                self.quaternion = [
                                    data_json['x'],
                                    data_json['y'],
                                    data_json['z'],
                                    data_json['w']
                                ]
                                self.position = np.array([
                                    data_json.get('pos_x', 0.0),
                                    data_json.get('pos_y', 0.0),
                                    data_json.get('pos_z', 0.0)
                                ])
                        except Exception as e:
                            print(f"Adatfeldolgozási hiba: {e} | Raw: {line}")
            except Exception as e:
                print(f"Socket error: {e}")
                break

    def update_camera_orientation(self):
        with self.lock:
            quat = self.quaternion.copy()
            # Normalizálás
            norm = np.linalg.norm(quat)
            if norm == 0:
                return
            quat = [q / norm for q in quat]

            # Rust → Python: [x, y, z, w], OpenGL camera "előre" = -Z
            rot = R.from_quat(quat)
            self.camera_front = rot.apply([0, 0, -1])    # nézeti irány
            self.camera_up = rot.apply([0, 1, 0])        # felfelé
            self.camera_pos = self.position + self.center  # világpozíció shiftelve a minta közepére

            # Normalizálás
            self.camera_front /= np.linalg.norm(self.camera_front)
            self.camera_up /= np.linalg.norm(self.camera_up)

    def process_input(self, delta_time):
        move_direction = np.zeros(3, dtype=np.float32)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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
        if keys[pygame.K_UP]:
            self.max_distance += 0.1
        if keys[pygame.K_DOWN]:
            self.max_distance = max(0.1, self.max_distance - 0.1)
        if keys[pygame.K_RIGHT]:
            self.alpha_value = min(1.0, self.alpha_value + 0.01)
        if keys[pygame.K_LEFT]:
            self.alpha_value = max(0.01, self.alpha_value - 0.01)
        if np.linalg.norm(move_direction) > 0:
            move_direction /= np.linalg.norm(move_direction)
            self.camera_pos += move_direction * self.movement_speed * delta_time
        return True

    def get_visible_points(self):
        directions = self.vertices - self.camera_pos
        distances = np.linalg.norm(directions, axis=1)
        directions_normalized = directions / distances[:, np.newaxis]
        dot = np.dot(directions_normalized, self.camera_front)
        mask = (distances < self.max_distance) & (dot > self.fov_cos)
        return self.vertices[mask], self.colors[mask]

    def generate_alpha_mesh(self, points, alpha=None):
        alpha = alpha if alpha is not None else self.alpha_value
        if len(points) < 10:
            self.mesh_triangles = None
            self.mesh_vertices = None
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            self.mesh_vertices = np.asarray(mesh.vertices)
            self.mesh_triangles = np.asarray(mesh.triangles)
        except Exception as e:
            self.mesh_vertices = None
            self.mesh_triangles = None

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.update_camera_orientation()
        cam_target = self.camera_pos + self.camera_front
        gluLookAt(*self.camera_pos, *cam_target, *self.camera_up)

        visible_points, _ = self.get_visible_points()
        current_hash = hash(visible_points.tobytes())
        if current_hash != self.last_visible_hash:
            self.generate_alpha_mesh(visible_points)
            self.last_visible_hash = current_hash

        glBegin(GL_POINTS)
        for point in visible_points:
            distance = np.linalg.norm(point - self.camera_pos)
            t = min(distance / self.max_distance, 1.0)
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

        if self.mesh_triangles is not None and self.mesh_vertices is not None:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.2, 0.6, 1.0, 0.3)
            glBegin(GL_TRIANGLES)
            for tri in self.mesh_triangles:
                for idx in tri:
                    vertex = self.mesh_vertices[idx]
                    distance = np.linalg.norm(vertex - self.camera_pos)
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
                    glColor4f(r, g, b, 1)
                    glVertex3fv(vertex)
            glEnd()
            glLineWidth(1.0)
            glColor3f(0.0, 0.0, 0.0)
            for tri in self.mesh_triangles:
                glBegin(GL_LINE_LOOP)
                for idx in tri:
                    glVertex3fv(self.mesh_vertices[idx])
                glEnd()
            glDisable(GL_BLEND)
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            delta_time = clock.tick(60) / 1000.0
            running = self.process_input(delta_time)
            self.render()
        self.running = False
        self.thread.join()
        self.socket.close()
        pygame.quit()

if __name__ == "__main__":
    viewer = PointCloudViewer("cave_sampled.ply", host='127.0.0.1', port=12345)
    viewer.run()
