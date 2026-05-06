import open3d as o3d
import time
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pandas as pd

class PointCloudViewerFromCSV:
    def __init__(self, point_cloud_file, imu_csv_file):
        # IMU adatok betöltése CSV-ből
        self.df = pd.read_csv(imu_csv_file, names=['timestamp', 'AccelX', 'AccelY', 'AccelZ', 
                                                    'GyroX', 'GyroY', 'GyroZ', 
                                                    'MagX', 'MagY', 'MagZ'])
        
        # Konvertálás numerikussá
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df.dropna()
        
        # Idő normalizálása
        self.df['time'] = self.df['timestamp'] - self.df['timestamp'].iloc[0]
        if self.df['time'].max() > 1000:
            self.df['time'] = self.df['time'] / 1000
        
        self.current_frame = 0
        self.total_frames = len(self.df)
        
        print(f"IMU adatok betöltve: {self.total_frames} minta")
        print(f"Időtartam: {self.df['time'].max():.2f} másodperc")
        
        # Kezdeti orientáció
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Pontfelhő betöltése
        self.pcd = o3d.io.read_point_cloud(point_cloud_file)
        self.vertices = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else np.ones_like(self.vertices) * 0.7
        
        # Pontfelhő középpontja
        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)
        
        # Kamera pozíció (fix)
        self.camera_pos = self.center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.max_distance = 10.0
        self.fov_cos = np.cos(np.radians(90))
        self.point_size = 3.0
        self.movement_speed = 1.0
        
        self.alpha_value = 0.6
        self.mesh_triangles = None
        self.mesh_vertices = None
        
        # Pygame inicializálás
        pygame.init()
        self.display = (1980, 1200)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.mouse.set_visible(True)
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glPointSize(self.point_size)
        
        self.running = True
        
        # Kvaternió a forgatáshoz
        self.quaternion_w = 1.0
        self.quaternion_x = 0.0
        self.quaternion_y = 0.0
        self.quaternion_z = 0.0
        
        # Kiegészítő szűrő az orientációhoz
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        self.filtered_yaw = 0.0
    
    def accel_to_orientation(self, accel_x, accel_y, accel_z):
        """Gyorsulásmérő adatokból orientáció számítása"""
        # Szűrés
        alpha = 0.8
        self.filtered_roll = alpha * self.filtered_roll + (1 - alpha) * np.arctan2(accel_y, accel_z)
        self.filtered_pitch = alpha * self.filtered_pitch + (1 - alpha) * np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2))
        
        return self.filtered_roll, self.filtered_pitch
    
    def update_from_csv(self):
        """Következő IMU minta betöltése CSV-ből"""
        if self.current_frame < self.total_frames:
            row = self.df.iloc[self.current_frame]
            
            # Gyorsulás adatok
            accel_x = row['AccelX']
            accel_y = row['AccelY']
            accel_z = row['AccelZ']
            
            # Orientáció számítása gyorsulásból
            roll, pitch = self.accel_to_orientation(accel_x, accel_y, accel_z)
            
            # Giroszkóp adatok (szögsebesség)
            gyro_x = row['GyroX']
            gyro_y = row['GyroY']
            gyro_z = row['GyroZ']
            
            # Kvaternió frissítése
            dt = 0.02  # Feltételezett időlépés (20ms)
            
            # Kvaternió számítása a giroszkóp adatokból
            # Egyszerű Euler-szöges megközelítés
            self.roll += gyro_x * dt
            self.pitch += gyro_y * dt
            self.yaw += gyro_z * dt
            
            # Kvaternió a forgatási mátrixhoz
            self.quaternion_to_matrix()
            
            self.current_frame += 1
            
            # Debug
            if self.current_frame % 100 == 0:
                print(f"Frame {self.current_frame}/{self.total_frames}, "
                      f"Roll: {np.degrees(self.roll):.1f}°, "
                      f"Pitch: {np.degrees(self.pitch):.1f}°, "
                      f"Yaw: {np.degrees(self.yaw):.1f}°")
    
    def quaternion_to_matrix(self):
        """Euler-szögekből kvaternió számítása"""
        # Egyszerűbb megközelítés: használjuk a forgatási mátrixot
        R = np.array([
            [np.cos(self.yaw)*np.cos(self.pitch), 
             np.cos(self.yaw)*np.sin(self.pitch)*np.sin(self.roll) - np.sin(self.yaw)*np.cos(self.roll),
             np.cos(self.yaw)*np.sin(self.pitch)*np.cos(self.roll) + np.sin(self.yaw)*np.sin(self.roll)],
            [np.sin(self.yaw)*np.cos(self.pitch),
             np.sin(self.yaw)*np.sin(self.pitch)*np.sin(self.roll) + np.cos(self.yaw)*np.cos(self.roll),
             np.sin(self.yaw)*np.sin(self.pitch)*np.cos(self.roll) - np.cos(self.yaw)*np.sin(self.roll)],
            [-np.sin(self.pitch),
             np.cos(self.pitch)*np.sin(self.roll),
             np.cos(self.pitch)*np.cos(self.roll)]
        ])
        
        # Forgatási mátrixból kvaternió
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            self.quaternion_w = 0.25 * S
            self.quaternion_x = (R[2,1] - R[1,2]) / S
            self.quaternion_y = (R[0,2] - R[2,0]) / S
            self.quaternion_z = (R[1,0] - R[0,1]) / S
        else:
            # További esetek kezelése...
            pass
    
    def update_camera_orientation(self):
        """Kamera orientáció frissítése kvaternióból"""
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
        R = np.array([
            [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
            [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
            [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
        ])
        
        self.camera_front = R[:, 2]
        self.camera_up = R[:, 1]
        
        self.camera_front /= np.linalg.norm(self.camera_front)
        self.camera_up /= np.linalg.norm(self.camera_up)
    
    def process_input(self, delta_time):
        """Bevitel kezelése"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    # Szünet/indítás
                    pass
                elif event.key == pygame.K_r:
                    # Vissza az elejére
                    self.current_frame = 0
                    self.roll = 0.0
                    self.pitch = 0.0
                    self.yaw = 0.0
                    print("Vissza az elejére")
        
        keys = pygame.key.get_pressed()
        
        # Mozgás
        move_direction = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_w]:
            move_direction += self.camera_front
        if keys[pygame.K_s]:
            move_direction -= self.camera_front
        if keys[pygame.K_a]:
            right = np.cross(self.camera_front, self.camera_up)
            right /= np.linalg.norm(right)
            move_direction -= right
        if keys[pygame.K_d]:
            right = np.cross(self.camera_front, self.camera_up)
            right /= np.linalg.norm(right)
            move_direction += right
        
        if np.linalg.norm(move_direction) > 0:
            move_direction /= np.linalg.norm(move_direction)
            self.camera_pos += move_direction * self.movement_speed * delta_time
        
        return True
    
    def get_visible_points(self):
        """Látható pontok kiválasztása"""
        directions = self.vertices - self.camera_pos
        distances = np.linalg.norm(directions, axis=1)
        directions_normalized = directions / distances[:, np.newaxis]
        
        dot = np.dot(directions_normalized, self.camera_front)
        mask = (distances < self.max_distance) & (dot > self.fov_cos)
        
        return self.vertices[mask], self.colors[mask]
    
    def generate_alpha_mesh(self, points):
        """Alpha shape generálás"""
        if len(points) < 10:
            self.mesh_triangles = None
            self.mesh_vertices = None
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, self.alpha_value)
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            
            self.mesh_vertices = np.asarray(mesh.vertices)
            self.mesh_triangles = np.asarray(mesh.triangles)
        except Exception as e:
            print("Hiba az alpha shape generáláskor: ", e)
            self.mesh_vertices = None
            self.mesh_triangles = None
    
    def render(self):
        """Megjelenítés"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        self.update_camera_orientation()
        
        cam_target = self.camera_pos + self.camera_front
        gluLookAt(*self.camera_pos, *cam_target, *self.camera_up)
        
        visible_points, _ = self.get_visible_points()
        
        # Alpha mesh generálás
        self.generate_alpha_mesh(visible_points)
        
        # Pontok kirajzolása
        glBegin(GL_POINTS)
        for point in visible_points:
            distance = np.linalg.norm(point - self.camera_pos)
            t = min(distance / self.max_distance, 1.0)
            
            # Színezés távolság alapján
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
        
        # Mesh kirajzolása
        if self.mesh_triangles is not None and self.mesh_vertices is not None:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Átlátszó háromszögek
            glColor4f(0.2, 0.6, 1.0, 0.3)
            glBegin(GL_TRIANGLES)
            for tri in self.mesh_triangles:
                for idx in tri:
                    glVertex3fv(self.mesh_vertices[idx])
            glEnd()
            
            # Körvonalak
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
        """Fő ciklus"""
        clock = pygame.time.Clock()
        last_update = time.time()
        running = True
        
        while running:
            delta_time = clock.tick(60) / 1000.0
            running = self.process_input(delta_time)
            
            # IMU adatok frissítése (kb. 50 Hz)
            if time.time() - last_update > 0.02:
                self.update_from_csv()
                last_update = time.time()
            
            self.render()
            
            # Ha vége a fájlnak, újrakezdés
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
                self.roll = 0.0
                self.pitch = 0.0
                self.yaw = 0.0
                print("Újraindítás")
        
        pygame.quit()

# Használat
if __name__ == "__main__":
    viewer = PointCloudViewerFromCSV(
        point_cloud_file="centered_sampled20000.ply",
        imu_csv_file="exp_2_imu.csv"
    )
    viewer.run()