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

class PointCloudViewer:
    def __init__(self, point_cloud_file, host='192.168.249.52', port=12345):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        print("Connected to AR glasses server")
        
        self.roll = 0.0 #orientációs adatok fokban
        self.pitch = 0.0
        self.yaw = 0.0
        self.filtered_roll = 0.0 #aluláteresztős szűrővel simított értékek
        self.filtered_pitch = 0.0
        self.filtered_yaw = 0.0
        self.lock = threading.Lock() #hozzáférés a megosztott változókhoz
        
        self.running = True
        self.thread = threading.Thread(target=self.receive_data)
        self.thread.daemon = True
        self.thread.start()

        self.pcd = o3d.io.read_point_cloud(point_cloud_file) #pontfelhő betöltése
        self.vertices = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else np.ones_like(self.vertices) * 0.7 #ha nincsenek színek, akkor alapértelmezett szürke színt használ

        #pontfelhő statisztikák készítése
        self.center = np.mean(self.vertices, axis=0) #pontfelhő középpont
        self.bounds_min = np.min(self.vertices, axis=0) #pontfelhő min határ
        self.bounds_max = np.max(self.vertices, axis=0) #pontfelhő max határ
        print(f"Pontfelhő középpontja: {self.center}")
        print(f"Kiterjedése: min={self.bounds_min}, max={self.bounds_max}")

        #kamera kezdő pozíciója
        self.camera_pos = self.center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.max_distance = 5.0 #maximális megjelenítési távolság
        self.fov_cos = np.cos(np.radians(90)) #látómező értéke
        self.point_size = 3.0 #pontok mérete
        self.movement_speed = 1.0 #kamera mozgási sebessége

        self.alpha_value = 0.6 #mennyire legyen sima a felület
        #ezek azért vannak, hogy ne számoljuk újra minden képkockában
        self.last_visible_hash = None
        self.mesh_triangles = None
        self.mesh_vertices = None

        pygame.init() #létrehozza a megjelenítő ablakot
        self.display = (1980, 1200)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL) #az OPENGL miatt tudunk navigálni az ablakban
        pygame.mouse.set_visible(True) #egérkurzor láthatósága

        glMatrixMode(GL_PROJECTION) #3D világ 2D-be vetítése
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 100.0) # 45=látómező szöge, 0.1 és 100 pedig a minden ami közelebb van, mint 0.1 vagy távolabb, mint 100, az biztosan nem látható
        glMatrixMode(GL_MODELVIEW) #objektudom elhelyezése a világban (pl. kamera)
        glEnable(GL_DEPTH_TEST) #csak azok a pixelek rajzolódnak ki, amik közelebb vannak a nézőponthoz
        glPointSize(self.point_size) #beállítja a pontok megadott méretét

        self.quaternion_w = 1.0
        self.quaternion_x = 0.0
        self.quaternion_y = 0.0
        self.quaternion_z = 0.0

    def receive_data(self):
        buffer = ""
        while self.running:
            try:
                data = self.socket.recv(1024).decode()
                if not data:
                    print("No data received, connection might be closed")
                    break
                    
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            quat_data = json.loads(line)
                            with self.lock:
                                # Frissítjük a kvaternió értékeket
                                self.quaternion_w = quat_data['w']
                                self.quaternion_x = quat_data['x']
                                self.quaternion_y = quat_data['y']
                                self.quaternion_z = quat_data['z']
                        except json.JSONDecodeError as e:
                            print(f"Invalid data: {line}, Error: {e}")
                        except KeyError as e:
                            print(f"Missing key in data: {e}, Line: {line}")
            except Exception as e:
                print(f"Socket error: {e}")
                break

    def update_camera_orientation(self):
        with self.lock:
            q_w = self.quaternion_w
            q_x = self.quaternion_x
            q_y = self.quaternion_y
            q_z = self.quaternion_z
            
            # Normalizáljuk a kvaterniót
            norm = np.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)
            q_w /= norm
            q_x /= norm
            q_y /= norm
            q_z /= norm
            
            q_x = -q_x
            q_y = q_y
            q_z = -q_z
            # w marad változatlan
            
            # Kvaternióból forgatási mátrix
            def quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z):
                R = np.array([
                    [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
                    [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
                    [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
                ])
                return R

            # Kamera irányok a mátrixból
            R = quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)
            self.camera_front = R[:, 2]  # Z tengely = előre
            self.camera_up = R[:, 1]     # Y tengely = felfelé
            
            # Normalizálás
            self.camera_front /= np.linalg.norm(self.camera_front)
            self.camera_up /= np.linalg.norm(self.camera_up)

    def process_input(self, delta_time):
        running = True
        move_direction = np.zeros(3, dtype=np.float32)

        for event in pygame.event.get(): #kilépés
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
            self.camera_pos += move_direction * self.movement_speed * delta_time

        return running

    def get_visible_points(self):
        directions = self.vertices - self.camera_pos #vektorok pozíciójának számolása a kamera pozíciójából
        distances = np.linalg.norm(directions, axis=1) #euklideszi távolság számolása, tehát, hogy milyen messze vannak a pontok a kamerától
        directions_normalized = directions / distances[:, np.newaxis] #egységvektorrá alakítás

        dot = np.dot(directions_normalized, self.camera_front) #skaláris szorzat a normalizált irányvektorok és a kamera nézeti iránya között
        mask = (distances < self.max_distance) & (dot > self.fov_cos) #láthatósági feltétel, a legyen közelebb, mint a max távolság, és legyen a látómezőben

        return self.vertices[mask], self.colors[mask] #visszaadott pontok és színe
    
    def generate_alpha_mesh(self, points, alpha=None):
        alpha = alpha if alpha is not None else self.alpha_value #a generált felület simaságát határozza meg, ha nincs megadva más, akkor a megadott alpha value alapján
        if len(points)<10: #random feltétel, hogy ha kevés a megjelenő pont, akkor nem generál felületet
            self.mesh_triangles = None
            self.mesh_vertices = None
        
        pcd = o3d.geometry.PointCloud() #pontfelhő létrehozása
        pcd.points = o3d.utility.Vector3dVector(points)
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha) #mesh készítés
            #feldolgozandó adatmennyiség csökkentése
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            
            #itt mentjük az eredményeket, lokáliskoordinátákat
            self.mesh_vertices = np.asarray(mesh.vertices) 
            self.mesh_triangles = np.asarray(mesh.triangles)
        except Exception as e:
            print("Hiba az alpha shape generáláskor: ",e) #pl ha túl kicsi al alpha
            self.mesh_vertices = None
            self.mesh_triangles = None

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) #törli a szín és a mélységbuffert mindig, ergo az előző képkocka információit

        #betölti az egységmátrixot és nullázza az előző transzformációt
        glMatrixMode(GL_MODELVIEW) 
        glLoadIdentity()

        self.update_camera_orientation() #irányok fetchelése

        print(f"Camera Front: {self.camera_front}, Camera Up: {self.camera_up}")
        
        cam_target = self.camera_pos + self.camera_front #célpont
        gluLookAt(*self.camera_pos, *cam_target, *self.camera_up) #beállítja a kamera pozícióját

        visible_points, _ = self.get_visible_points() #látható pontok fetch

        current_hash = hash(visible_points.tobytes()) #láthatóság számítása
        if current_hash != self.last_visible_hash: #csak akkor generál új mesht ha változott a pontok halmaza egy adott területen
            self.generate_alpha_mesh(visible_points)
            self.last_visible_hash = current_hash

        glBegin(GL_POINTS) #minden pontot lerajzol külön
        for point in visible_points: #színátmenetes távolság
            distance = np.linalg.norm(point - self.camera_pos)
            t = min(distance / self.max_distance, 1.0)

            if t < 0.5: #közeli pontok: pirosból zöld átmenet
                r = 1.0 - 2 * t
                g = 2 * t
                b = 0.0
            else: #távoli pontok: zöldből kék átmenet
                t2 = (t - 0.5) * 2
                r = 0.0
                g = 1.0 - t2
                b = t2

            glColor3f(r, g, b)
            glVertex3fv(point)
        glEnd()

        # Alpha shape 
        if self.mesh_triangles is not None and self.mesh_vertices is not None:
            glEnable(GL_BLEND) #színkeverés engedélyezése
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) #forrás szín alpha komponensével szoroz
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
            glLineWidth(1.0) #él vastagság
            glColor3f(0.0, 0.0, 0.0)  #fekete élek
            for tri in self.mesh_triangles:
                glBegin(GL_LINE_LOOP)
                for idx in tri:
                    glVertex3fv(self.mesh_vertices[idx])
                glEnd()
            glDisable(GL_BLEND) #színkeverés


        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        last_print_time = time.time()
        running = True

        while running:
            delta_time = clock.tick(60) / 1000.0
            running = self.process_input(delta_time)
            self.render()

            if time.time() - last_print_time > 0.05: #fél másodpercenként frissül a kimenet
                visible_points, _ = self.get_visible_points()
                with self.lock:
                    print(f"Orientáció - Roll (billenés): {self.roll:.2f}, Pitch (fel-le): {self.pitch:.2f}, Yaw (jobbra-balra): {self.yaw:.2f}")
                    print(f"Látható pontok: {len(visible_points)}")
                    #print(f"Látható pontok: {len(visible_points)}, Pozíció: {self.camera_pos}")
                last_print_time = time.time()

        self.running = False #jelet küld az adatfogadónak, ha leáll
        self.thread.join()
        self.socket.close()
        pygame.quit()

if __name__ == "__main__":
    viewer = PointCloudViewer("cave_sampled.ply", host='127.0.0.1')
    viewer.run()