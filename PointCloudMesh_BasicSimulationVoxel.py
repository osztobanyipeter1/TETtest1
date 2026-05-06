import open3d as o3d
import time
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


class PointCloudViewer:
    def __init__(self, point_cloud_file, voxel_size=0.2):
        self.pcd = o3d.io.read_point_cloud(point_cloud_file)
        self.voxel_size = voxel_size
        
        # VOXEL GRID PREPROCESSING - CSAK FOGLALT VOXEL KÖZÉPPONTJAI + MAX INTENZITÁS
        print("Voxel grid létrehozása...")
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size=voxel_size)
        voxels = voxel_grid.get_voxels()
        print(f"Foglalt voxelok száma: {len(voxels)}")

        self.voxel_centers = []
        self.voxel_intensities = []  # ÚJ: max intenzitás voxel-enként (0-255 -> 0-1)

        # Intensity ellenőrzés **HELYES MÓD** Open3D-ben
        has_intensity = False
        try:
            # Próbáld meg elérni - ha nincs, KeyError dob
            test_intensities = np.asarray(self.pcd.point["intensity"])
            has_intensity = True
            print(f"✅ Intenzitás megtalálva: min={test_intensities.min():.0f}, max={test_intensities.max():.0f} (uchar 0-255)")
        except KeyError:
            print("❌ Nincs 'intensity' attribútum, fallback szürke + távolság színezésre")
        except Exception as e:
            print(f"❌ Intensity ellenőrzés hiba: {e}, fallback")

        for vox in voxels:
            center = voxel_grid.get_voxel_center_coordinate(vox.grid_index)
            self.voxel_centers.append(center)
            
            intensity = 0.0  # Default fallback
            
            if has_intensity:
                try:
                    # Voxel bounding box
                    gi = vox.grid_index
                    min_bound = np.array([gi[0]*voxel_size, gi[1]*voxel_size, gi[2]*voxel_size])
                    max_bound = min_bound + voxel_size
                    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                    voxel_points = self.pcd.crop(bbox)
                    if len(voxel_points.points) > 0:
                        voxel_intensities = np.asarray(voxel_points.point["intensity"])
                        intensity = voxel_intensities.max() / 255.0  # Normalizálás 0-1
                except:
                    intensity = 0.0  # Bármi hiba -> fallback
            
            self.voxel_intensities.append(intensity)

        self.vertices = np.asarray(self.voxel_centers)
        self.intensities = np.array(self.voxel_intensities)

        # Globális normalizálás
        int_min, int_max = self.intensities.min(), self.intensities.max()
        if int_max > int_min:
            self.intensities = (self.intensities - int_min) / (int_max - int_min)
        else:
            self.intensities = np.full(len(self.vertices), 0.5)

        self.colors = self.intensity_to_color(self.intensities)

        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)

        print(f"Voxelizált pontfelhő: {len(self.vertices)} pont")
        print(f"Intenzitás tartomány: {self.intensities.min():.3f} - {self.intensities.max():.3f}")
        print(f"Középpont: {self.center}")
        print(f"Kiterjedés: min={self.bounds_min}, max={self.bounds_max}")


        # Kamera (változatlan)
        self.camera_pos = self.center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_front = (self.center - self.camera_pos)
        self.camera_front /= np.linalg.norm(self.camera_front)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0

        self.max_distance = 20.0
        self.fov_cos = np.cos(np.radians(60))
        self.point_size = 4.0
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

    def intensity_to_color(self, intensities):
        """Intenzitás -> szín: nagy= sötétkék (0,0,1), kicsi= sárga (1,1,0)"""
        # Invert: magas intenzitás -> kék, alacsony -> sárga
        t = 1.0 - intensities  # 0 (magas int) -> sárga, 1 (alacsony int) -> kék? VÁRJ, fordítva!
        t = intensities  # magas int -> magas t -> kék
        
        r = 1.0 - t  # magas t -> alacsony r (kék)
        g = 1.0 - t * 0.5  # enyhe zöld csillapítás
        b = t  # magas t -> magas kék
        
        colors = np.zeros((len(intensities), 3))
        colors[:, 0] = r
        colors[:, 1] = g
        colors[:, 2] = b
        return colors

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

        return self.vertices[mask], self.colors[mask]  # Most colors intenzitás alapú!

    def generate_alpha_mesh(self, points, alpha=None):
        alpha = alpha if alpha is not None else self.alpha_value
        if len(points) < 10:
            self.mesh_triangles = None
            self.mesh_vertices = None
            return

        # Mesh-hez színek: eredeti voxel színek (intenzitás alapú)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors[:len(points)])
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

        visible_points, visible_colors = self.get_visible_points()  # Intenzitás színek!

        current_hash = hash(visible_points.tobytes())
        if current_hash != self.last_visible_hash:
            self.generate_alpha_mesh(visible_points)
            self.last_visible_hash = current_hash

        # VOXEL PONTok kirajzolása (INTENZITÁS ALAPÚ FIX SZÍNEKKEL)
        glBegin(GL_POINTS)
        for i, point in enumerate(visible_points):
            # FIX szín intenzitás alapján - nincs távolság moduláció!
            if len(visible_colors) > i:
                col = visible_colors[i]
                glColor3f(col[0], col[1], col[2])
            else:
                glColor3f(0.7, 0.7, 0.7)
            glVertex3fv(point)
        glEnd()

        # Alpha shape mesh (intenzitás színekkel)
        if self.mesh_triangles is not None and self.mesh_vertices is not None:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glBegin(GL_TRIANGLES)
            for tri in self.mesh_triangles:
                for idx in tri:
                    vertex = self.mesh_vertices[idx]
                    # Mesh-hez is távolság moduláció + intenzitás szín (de mivel nincs pontos szín, közelítő)
                    distance = np.linalg.norm(vertex - self.camera_pos)
                    t = min(distance / self.max_distance, 1.0)
                    base_col = np.array([0.2, 0.6, 1.0])  # Alap kékes
                    col = base_col * (1 - 0.3 * t)
                    glColor4f(col[0], col[1], col[2], self.alpha_value)
                    glVertex3fv(vertex)
            glEnd()
            
            # Élek fekete vonalakkal
            glDisable(GL_BLEND)
            glLineWidth(1.0)
            glColor3f(0.0, 0.0, 0.0)
            glEnableClientState(GL_VERTEX_ARRAY)
            for tri in self.mesh_triangles:
                glBegin(GL_LINE_LOOP)
                for idx in tri:
                    glVertex3fv(self.mesh_vertices[idx])
                glEnd()
            glDisableClientState(GL_VERTEX_ARRAY)

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
    viewer = PointCloudViewer("newship.ply", voxel_size=0.2)
    viewer.run()
