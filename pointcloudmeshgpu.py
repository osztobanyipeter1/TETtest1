import open3d as o3d
import time
import numpy as np
import cupy as cp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class PointCloudViewer:
    def __init__(self, point_cloud_file):
        try:
            cp.cuda.Device(0).compute_capability
            print("GPU ok:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
        except cp.cuda.runtime.CUDARuntimeError:
            print("GPU NOT OK")
            exit(1)

        self.pcd = o3d.io.read_point_cloud(point_cloud_file)
        np_vertices = np.asarray(self.pcd.points)
        np_colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else np.ones_like(np_vertices) * 0.7

        self.vertices = cp.asarray(np_vertices)
        self.colors = cp.asarray(np_colors)

        self.center = cp.mean(self.vertices, axis=0).get()
        self.bounds_min = cp.min(self.vertices, axis=0).get()
        self.bounds_max = cp.max(self.vertices, axis=0).get()

        print(f"Pontfelhő középpontja: {self.center}")
        print(f"Kiterjedése: min={self.bounds_min}, max={self.bounds_max}")

        self.camera_pos = self.center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_front = (self.center - self.camera_pos)
        self.camera_front /= np.linalg.norm(self.camera_front)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0

        self.max_distance = 6.0
        self.fov_cos = np.cos(np.radians(60))
        self.point_size = 3.0
        self.movement_speed = 1.0
        self.mouse_sensitivity = 0.1

        self.alpha_value = 0.6
        self.last_visible_hash = None
        self.mesh_triangles = None
        self.mesh_vertices = None

        pygame.init()
        self.display = (1500, 720)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glPointSize(self.point_size)

        pygame.mouse.get_rel()

    def update_camera_vectors(self):
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ], dtype=np.float32)
        self.camera_front = front / np.linalg.norm(front)

    def process_input(self, delta_time):
        running = True
        move_direction = np.zeros(3, dtype=np.float32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
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

        dx, dy = pygame.mouse.get_rel()
        if dx != 0 or dy != 0:
            self.yaw += dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self.update_camera_vectors()

        return running

    def get_visible_points(self):
        camera_pos_gpu = cp.asarray(self.camera_pos, dtype=cp.float32)
        camera_front_gpu = cp.asarray(self.camera_front, dtype=cp.float32)

        directions = self.vertices - camera_pos_gpu
        distances = cp.linalg.norm(directions, axis=1)
        directions_normalized = directions / distances[:, cp.newaxis]

        dot = directions_normalized @ camera_front_gpu
        mask = (distances < self.max_distance) & (dot > self.fov_cos)

        return cp.asnumpy(self.vertices[mask]), cp.asnumpy(self.colors[mask])

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
            mesh.remove_non_manifold_edges()

            self.mesh_vertices = np.asarray(mesh.vertices)
            self.mesh_triangles = np.asarray(mesh.triangles)
        except Exception as e:
            print("Hiba az alpha shape generálásakor:", e)
            self.mesh_vertices = None
            self.mesh_triangles = None

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

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
                t2 = t * 2
                r, g, b = t2, 1.0 - t2, 1.0
            else:
                t2 = (t - 0.5) * 2
                r, g, b = 1.0, 1.0 - t2, 1.0 - t2

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
                        r, g, b = t2, 1.0 - t2, 1.0
                    else:
                        t2 = (t - 0.5) * 2
                        r, g, b = 1.0, 1.0 - t2, 1.0 - t2
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
        self.update_camera_vectors()
        running = True
        last_print_time = time.time()

        while running:
            delta_time = clock.tick(60) / 1000.0
            self.fps = clock.get_fps()
            running = self.process_input(delta_time)
            self.render()

            if time.time() - last_print_time > 0.5:
                visible_points, _ = self.get_visible_points()
                print(f"Látható pontok: {len(visible_points)}, FPS: {int(self.fps)}, Pozíció: {self.camera_pos}, Alpha: {self.alpha_value:.2f}")
                last_print_time = time.time()

        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        pygame.quit()

if __name__ == "__main__":
    print("""
    Vezérlés:
    - W, A, S, D: Mozgás
    - SPACE, LSHIFT: Fel / Le
    - Egér: Kamera forgatás
    - NYILAK FEL/LE: max_distance növelése/csökkentése
    - ESC: Kilépés
    """)
    viewer = PointCloudViewer("cave_sampled3.ply")
    viewer.run()
