import open3d as o3d
import time
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


class PointCloudViewer:
    def __init__(self, point_cloud_file, voxel_size=0.1):
        self.pcd = o3d.io.read_point_cloud(point_cloud_file)
        self.voxel_size = voxel_size

        # Voxel grid preprocessing - csak a foglalt voxelok középpontjai
        print("Voxel grid létrehozása...")
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size=voxel_size)
        voxels = voxel_grid.get_voxels()
        print(f"Foglalt voxelok száma: {len(voxels)}")

        self.voxel_centers = []
        self.voxel_colors = []
        for vox in voxels:
            center = voxel_grid.get_voxel_center_coordinate(vox.grid_index)
            self.voxel_centers.append(center)
        
        self.vertices = np.asarray(self.voxel_centers)
        self.colors = np.ones_like(self.vertices) * 0.7  # Alap szín, ha nincs szín

        # Színek átlagolása opcionálisan (ha van szín a pcd-ben)
        if self.pcd.has_colors():
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size=voxel_size)
            # Egyszerűbb: voxel downsample-t használjuk színre
            down_pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
            self.colors = np.asarray(down_pcd.colors)

        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)

        print(f"Voxelizált pontfelhő középpontja: {self.center}")
        print(f"Kiterjedése: min={self.bounds_min}, max={self.bounds_max}")
        print(f"Pontok száma voxel downsample után: {len(self.vertices)}")

        self.camera_pos = self.center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_front = (self.center - self.camera_pos)
        self.camera_front /= np.linalg.norm(self.camera_front)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0

        self.max_distance = 20.0
        self.fov_cos = np.cos(np.radians(45))
        self.point_size = 5.0  # Nagyobb pontméret voxel pontokhoz
        self.movement_speed = 1.0
        self.mouse_sensitivity = 0.1

        pygame.init()
        self.display = (1280, 900)
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
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    # Újratöltés nagyobb voxel_size-szal teszteléshez
                    return 'reload'

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

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cam_target = self.camera_pos + self.camera_front
        gluLookAt(*self.camera_pos, *cam_target, *self.camera_up)

        visible_points, visible_colors = self.get_visible_points()

        glBegin(GL_POINTS)
        for i, point in enumerate(visible_points):
            distance = np.linalg.norm(point - self.camera_pos)
            t = min(distance / self.max_distance, 1.0)

            # Távolság alapú színátmenet (zöld-sárga-piros)
            if t < 0.5:
                r = 1.0 - 2 * t
                g = 2 * t
                b = 0.0
            else:
                t2 = (t - 0.5) * 2
                r = 0.0
                g = 1.0 - t2
                b = t2

            # Ha van voxel szín, azt használd, különben távolság színt
            if hasattr(self, 'colors') and len(visible_colors) > i:
                col = visible_colors[i]
                glColor3f(col[0], col[1], col[2])
            else:
                glColor3f(r, g, b)
            
            glVertex3fv(point)
        glEnd()

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        self.update_camera_vectors()
        running = True
        last_print_time = time.time()

        while running:
            delta_time = clock.tick(60) / 1000.0
            self.fps = clock.get_fps()
            
            input_result = self.process_input(delta_time)
            if input_result == False:
                break
            elif input_result == 'reload':
                break
                
            self.render()

            if time.time() - last_print_time > 0.5:
                visible_points, _ = self.get_visible_points()
                print(f"Látható voxel pontok: {len(visible_points)}, FPS: {int(self.fps)}, Pozíció: {self.camera_pos}")
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
    - R: Újratöltés (különböző voxel_size-szal teszteléshez)
    - ESC: Kilépés
    
    Megjegyzés: voxel_size=0.1 a __init__-ben állítható!
    """)
    viewer = PointCloudViewer("sonar_imu_output1.ply", voxel_size=0.2)  # Itt állítsd a kívánt voxel méretet
    viewer.run()
