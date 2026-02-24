import open3d as o3d
import time
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


class PointCloudViewer:
    def __init__(self, point_cloud_file, voxel_size=0.05):
        self.pcd = o3d.io.read_point_cloud(point_cloud_file)
        self.voxel_size = voxel_size
        
        # VOXEL GRID PREPROCESSING - CSAK FOGLALT VOXEL KÖZÉPPONTJAI
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
        
        # Színek: ha van szín a PLY-ben, akkor voxel downsample-ból
        if self.pcd.has_colors():
            down_pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
            self.colors = np.asarray(down_pcd.colors)
        else:
            self.colors = np.ones_like(self.vertices) * 0.7
        
        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)

        print(f"Voxelizált pontfelhő: {len(self.vertices)} pont")
        print(f"Középpont: {self.center}")
        print(f"Kiterjedés: min={self.bounds_min}, max={self.bounds_max}")

        # Kamera
        self.camera_pos = self.center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_front = (self.center - self.camera_pos)
        self.camera_front /= np.linalg.norm(self.camera_front)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0

        self.max_distance = 20.0
        self.fov_cos = np.cos(np.radians(60))
        self.point_size = 4.0  # Nagyobb voxel pontokhoz
        self.movement_speed = 1.0
        self.mouse_sensitivity = 0.1

        self.alpha_value = 0.5
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
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    return 'reload'  # Újratöltés különböző voxel_size-szal

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
        pcd.colors = o3d.utility.Vector3dVector(self.colors[:len(points)])  # Voxel színek hozzáadása
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

        visible_points, visible_colors = self.get_visible_points()

        current_hash = hash(visible_points.tobytes())
        if current_hash != self.last_visible_hash:
            self.generate_alpha_mesh(visible_points)
            self.last_visible_hash = current_hash

        # VOXEL PONTok kirajzolása (voxel színekkel)
        glBegin(GL_POINTS)
        for i, point in enumerate(visible_points):
            distance = np.linalg.norm(point - self.camera_pos)
            t = min(distance / self.max_distance, 1.0)

            # Voxel szín használata távolság helyett
            if len(visible_colors) > i:
                col = visible_colors[i]
                # Távolság moduláció
                r, g, b = col[0], col[1], col[2]
                r *= (1 - 0.3 * t)
                g *= (1 - 0.3 * t)
                b *= (1 - 0.3 * t)
            else:
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

        # Alpha shape mesh (voxel pontokon)
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

                    glColor4f(r, g, b, self.alpha_value) 
                    glVertex3fv(vertex)
            glEnd()
            
            # Élek fekete vonalakkal
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
            
            input_result = self.process_input(delta_time)
            if input_result == False:
                break
            elif input_result == 'reload':
                break
                
            self.render()

            if time.time() - last_print_time > 0.5:
                visible_points, _ = self.get_visible_points()
                print(f"Látható VOXEL pontok: {len(visible_points)}, FPS: {int(self.fps)}, "
                      f"Pozíció: {self.camera_pos}, Alpha: {self.alpha_value:.2f}, "
                      f"Voxel_size: {self.voxel_size}")
                last_print_time = time.time()

        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        pygame.quit()


if __name__ == "__main__":
    print("""
    Vezérlés:
    - W, A, S, D: Mozgás          - NYILAK FEL/LE: max_distance
    - SPACE/LSHIFT: Fel/Le        - BAL/JOBB NYIL: alpha_value
    - Egér: Kamera forgatás       - R: ÚJRA TÖLTÉS (másik voxel_size)
    - ESC: Kilépés
    
    Voxel_size a hívásnál: PointCloudViewer("file.ply", voxel_size=0.05)
    """)
    viewer = PointCloudViewer("centered.ply", voxel_size=0.5)
    viewer.run()
