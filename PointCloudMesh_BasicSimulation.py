import time
import numpy as np
import open3d as o3d
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *


class PointCloudViewer:

    def __init__(self, point_cloud_file, scale_factor=0.5):
        self.pcd = o3d.io.read_point_cloud(point_cloud_file)
        self.vertices = np.asarray(self.pcd.points)
        self.vertices = self.vertices * scale_factor
        self.colors = (
            np.asarray(self.pcd.colors)
            if self.pcd.has_colors()
            else np.ones_like(self.vertices) * 0.7
        )
        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)

        print(f"Pontfelhő középpontja: {self.center}")
        print(f"Kiterjedése: min={self.bounds_min}, max={self.bounds_max}")

        self.camera_pos = self.center + np.array(
            [0.0, 0.0, 5.0], dtype=np.float32
        )
        self.camera_front = self.center - self.camera_pos
        self.camera_front /= np.linalg.norm(self.camera_front)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0

        self.max_distance = 20.0
        self.fov_cos = np.cos(np.radians(60))
        self.point_size = 3.0
        self.movement_speed = 1.0
        self.mouse_sensitivity = 0.1

        self.alpha_value = 0.1
        self.last_visible_hash = None
        self.mesh_triangles = None
        self.mesh_vertices = None
        self.mesh_colors = None

        pygame.init()
        self.display = (1920, 1080)
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
        front = np.array(
            [
                np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
                np.sin(np.radians(self.pitch)),
                np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            ],
            dtype=np.float32,
        )
        self.camera_front = front / np.linalg.norm(front)

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
            self.camera_pos += (
                move_direction * self.movement_speed * delta_time
            )

        dx, dy = pygame.mouse.get_rel()
        if dx != 0 or dy != 0:
            self.yaw += dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self.update_camera_vectors()

        return running

    def get_visible_points(self):
        directions = self.vertices - self.camera_pos
        distances = np.linalg.norm(directions, axis=1)
        directions_normalized = directions / distances[:, np.newaxis]

        dot = np.dot(directions_normalized, self.camera_front)
        mask = (distances < self.max_distance) & (dot > self.fov_cos)

        return self.vertices[mask], self.colors[mask]

    def generate_alpha_mesh(self, points, colors, alpha=None):
        alpha = alpha if alpha is not None else self.alpha_value
        if len(points) < 10:
            self.mesh_triangles = None
            self.mesh_vertices = None
            self.mesh_colors = None
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha
            )
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()

            self.mesh_vertices = np.asarray(mesh.vertices)
            self.mesh_triangles = np.asarray(mesh.triangles)

            if len(self.mesh_vertices) > 0:
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                mesh_colors = []
                for vertex in self.mesh_vertices:
                    _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
                    mesh_colors.append(colors[idx[0]])
                self.mesh_colors = np.array(mesh_colors)
            else:
                self.mesh_colors = None

        except Exception as e:
            print("Hiba az alpha shape generálásakor:", e)
            self.mesh_vertices = None
            self.mesh_triangles = None
            self.mesh_colors = None

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cam_target = self.camera_pos + self.camera_front
        gluLookAt(*self.camera_pos, *cam_target, *self.camera_up)

        visible_points, visible_colors = self.get_visible_points()

        current_hash = hash(visible_points.tobytes())
        if current_hash != self.last_visible_hash:
            self.generate_alpha_mesh(visible_points, visible_colors)
            self.last_visible_hash = current_hash

        glBegin(GL_POINTS)
        for point, color in zip(visible_points, visible_colors):
            glColor3fv(color)
            glVertex3fv(point)
        glEnd()

        if (
            self.mesh_triangles is not None
            and self.mesh_vertices is not None
            and self.mesh_colors is not None
        ):
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glBegin(GL_TRIANGLES)
            for tri in self.mesh_triangles:
                for idx in tri:
                    glColor3f(
                        self.mesh_colors[idx][0],
                        self.mesh_colors[idx][1],
                        self.mesh_colors[idx][2],
                    )
                    glVertex3fv(self.mesh_vertices[idx])
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
                print(
                    f"Látható pontok: {len(visible_points)}, FPS: {int(self.fps)}, Pozíció: {self.camera_pos}, Alpha: {self.alpha_value:.2f}"
                )
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
    viewer = PointCloudViewer("2ndkimenet.ply")
    viewer.run()