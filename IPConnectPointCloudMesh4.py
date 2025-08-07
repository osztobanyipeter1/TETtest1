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
import cv2
from ultralytics import YOLO
import logging
from collections import deque
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN



# YOLO warning elnyomása
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class ARPoseTrackingSystem:
    """
    Teljes AR pozíció követő rendszer vizuális-inerciális odometriával
    """
    
    def __init__(self):
        # 1. FÁZIS: ALAPVETŐ DETECTION ÉS TRACKING
        # Feature detection paraméterei (Shi-Tomasi) 
        self.feature_params = dict(
            maxCorners=500, #maximum 500 pontot talál
            qualityLevel=0.03, #csak a jó minőséfű pontokat tartja meg
            minDistance=5, #pontok közötti minimális távolság
            blockSize=7
        )
        
        # Lucas-Kanade optical flow paraméterei. 
        self.lk_params = dict(
            winSize=(21, 21), #kis ablakokat használ 21x21 pixeleset, amivel keresi, hogy hova került az ablak a kövi képkozkában
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # RANSAC paraméterek Essential Matrix becsléshez
        self.ransac_params = {
            'threshold': 1.0,
            'confidence': 0.99,
            'maxIters': 1000
        }
        
        # 2. FÁZIS: SKÁLA MEGHATÁROZÁS
        self.scale_history = deque(maxlen=10)
        self.position_3d = np.zeros(3)
        self.velocity_3d = np.zeros(3)
        self.scale_factor = 0.1  # Alapértelmezett skála. A scale_factor=0.1 szorzót használ, amit az objektumok méretváltozása alapján finomít.
        
        # 3. FÁZIS: BUNDLE ADJUSTMENT
        self.sliding_window_size = 10
        self.keyframes = deque(maxlen=self.sliding_window_size)
        self.frame_poses = deque(maxlen=self.sliding_window_size)
        self.landmarks_3d = {}
        self.landmark_observations = {}
        
        # 4. FÁZIS: DRIFT KORREKCIÓ
        self.bow_vocabulary = None
        self.loop_closure_threshold = 0.8
        self.pose_graph = []
        
        # 5. FÁZIS: ADATSTRUKTÚRÁK
        self.prev_frame = None
        self.prev_keypoints = None
        self.current_frame_id = 0
        self.tracking_state = "INITIALIZING"  # INITIALIZING, TRACKING, LOST
        
        # 6. FÁZIS: HIBAKEZELÉS
        self.min_features = 50
        self.max_reprojection_error = 2.0
        self.tracking_quality_threshold = 0.7
        
        # YOLO objektum tracking
        self.yolo_model = YOLO('yolov8s.pt')
        self.tracked_objects = {}
        self.object_id_counter = 0

class PointCloudViewer:
    def __init__(self, point_cloud_file, host='192.168.249.52', port=12345, camera_id=0):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        print("Connected to AR glasses server")
        
        self.lock = threading.Lock()
        self.running = True

        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.connect(('127.0.0.1', 12346))  # Másik port a pozíciód és orientációd küldésére
        self.data_socket_lock = threading.Lock()

        
        # Kamera inicializálás
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print("Kamera nem érhető el, vizuális tracking kikapcsolva")
            self.cap = None
        
        # AR Pose Tracking System inicializálás
        self.ar_tracker = ARPoseTrackingSystem()
        
        # Orientáció fogadó szál
        self.orientation_thread = threading.Thread(target=self.receive_orientation_data)
        self.orientation_thread.daemon = True
        self.orientation_thread.start()
        
        # Vizuális tracking szál
        if self.cap is not None:
            self.vision_thread = threading.Thread(target=self.visual_tracking_loop)
            self.vision_thread.daemon = True
            self.vision_thread.start()
        
        # Eredeti PointCloud beállítások
        self.pcd = o3d.io.read_point_cloud(point_cloud_file) #pontfelhő betöltése
        self.vertices = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors) if self.pcd.has_colors() else np.ones_like(self.vertices) * 0.7
        
        self.center = np.mean(self.vertices, axis=0)
        self.bounds_min = np.min(self.vertices, axis=0)
        self.bounds_max = np.max(self.vertices, axis=0)
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
        
        # Quaternion orientáció
        self.quaternion_w = 1.0
        self.quaternion_x = 0.0
        self.quaternion_y = 0.0
        self.quaternion_z = 0.0
        
        pygame.init() #létrehozza a megjelenítő ablakot
        self.display = (1980, 1200)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL) #az OPENGL miatt tudunk navigálni az ablakban
        pygame.mouse.set_visible(True) #egérkurzor láthatósága

        glMatrixMode(GL_PROJECTION) #3D világ 2D-be vetítése
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 100.0) # 45=látómező szöge, 0.1 és 100 pedig a minden ami közelebb van, mint 0.1 vagy távolabb, mint 100, az biztosan nem látható
        glMatrixMode(GL_MODELVIEW) #objektudom elhelyezése a világban (pl. kamera)
        glEnable(GL_DEPTH_TEST) #csak azok a pixelek rajzolódnak ki, amik közelebb vannak a nézőponthoz
        glPointSize(self.point_size) #beállítja a pontok megadott méretét

    def receive_orientation_data(self):
        """Orientáció fogadása a Rust szerverről"""
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
                                self.quaternion_w = quat_data['w']
                                self.quaternion_x = quat_data['x']
                                self.quaternion_y = quat_data['y']
                                self.quaternion_z = quat_data['z']
                        except (json.JSONDecodeError, KeyError) as e:
                            pass
            except Exception as e:
                print(f"Socket error: {e}")
                break

    def visual_tracking_loop(self): #képkockák folyamatos beolvasása
        """
        Fő vizuális tracking loop - az összes fázis implementálása
        """
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # 1. FÁZIS: FEATURE DETECTION ÉS TRACKING
            current_position = self.process_visual_frame(frame)
            
            # Pozíció frissítése a point cloud viewer számára
            if current_position is not None:
                with self.lock:
                    # AR tracking eredményének integrálása a kamera pozícióba
                    self.camera_pos = self.center + current_position
            
            time.sleep(1/30)  # 30 FPS

    def process_visual_frame(self, frame):
        """
        1. FÁZIS: Alapvető vizuális frame feldolgozás
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.ar_tracker.prev_frame is None:
            # Első frame inicializálás
            return self.initialize_tracking(gray, frame)
        else:
            # Folyamatos tracking
            return self.track_frame(gray, frame)

    def initialize_tracking(self, gray_frame, color_frame): #A kamera képét szürkeárnyalatossá alakítja, majd speciális pontokat keres. 
        """
        1.A) Shi-Tomasi feature detection inicializálás
        """
        # Feature detektálás
        features = cv2.goodFeaturesToTrack(
            gray_frame, 
            mask=None, 
            **self.ar_tracker.feature_params
        )
        
        if features is None or len(features) < self.ar_tracker.min_features:
            print("Nem található elegendő feature pont az inicializáláshoz")
            return None
        
        self.ar_tracker.prev_keypoints = features
        self.ar_tracker.prev_frame = gray_frame.copy()
        self.ar_tracker.tracking_state = "TRACKING"
        
        # YOLO objektum detektálás inicializálás
        self.detect_and_track_objects(color_frame)
        
        print(f"Inicializálás kész: {len(features)} feature pont")
        return np.zeros(3)

    def track_frame(self, gray_frame, color_frame): #Ez hasonlítja össze az előző és a jelenlegi kameraképet, követi az előző képkockában talált pontok hova mozognak. (jobbra-balra)
        """
        1.B) Lucas-Kanade optical flow tracking + Essential Matrix becslés
        """
        if self.ar_tracker.prev_keypoints is None:
            return self.initialize_tracking(gray_frame, color_frame)
        
        # 1.B.1) Lucas-Kanade optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.ar_tracker.prev_frame,
            gray_frame,
            self.ar_tracker.prev_keypoints,
            None,
            **self.ar_tracker.lk_params
        )
        
        # Jó pontok kiválasztása
        good_mask = (status.flatten() == 1) & (error.flatten() < 50)
        if np.sum(good_mask) < self.ar_tracker.min_features:
            print("Tracking elveszett - újrainicializálás")
            self.ar_tracker.tracking_state = "LOST"
            return self.initialize_tracking(gray_frame, color_frame)
        
        good_old = self.ar_tracker.prev_keypoints[good_mask]
        good_new = next_points[good_mask]
        
        # 1.B.2) Essential Matrix becslés RANSAC-cal
        motion_vector = self.estimate_camera_motion(good_old, good_new)
        
        # 2. FÁZIS: Skála becslés és pozíció frissítés
        scaled_motion = self.estimate_scale_and_update_position(motion_vector)
        
        # 3. FÁZIS: Bundle Adjustment (egyszerűsített)
        self.update_bundle_adjustment(good_old, good_new, scaled_motion)
        
        # 4. FÁZIS: Loop Closure Detection (egyszerűsített)
        self.detect_loop_closure(gray_frame)
        
        # YOLO objektum tracking
        self.detect_and_track_objects(color_frame)
        
        # 5. FÁZIS: Következő frame előkészítése
        self.ar_tracker.prev_frame = gray_frame.copy()
        self.ar_tracker.prev_keypoints = good_new.reshape(-1, 1, 2)
        self.ar_tracker.current_frame_id += 1
        
        return scaled_motion

    def estimate_camera_motion(self, prev_points, curr_points):
        """
        1.B.2) Essential Matrix becslés 5-pontos algoritmussal
        """
        if len(prev_points) < 8:
            return np.zeros(3)
        
        try:
            # Essential matrix becslés
            E, mask = cv2.findEssentialMat( # kamera mozgását (forgatás + elmozdulás) számítja ki két képkocka közötti pont-megfeleltetésekből
                curr_points, prev_points,
                focal=1.0, pp=(0, 0),
                method=cv2.RANSAC,
                prob=self.ar_tracker.ransac_params['confidence'],
                threshold=self.ar_tracker.ransac_params['threshold']
            )
            
            if E is None:
                return np.zeros(3)
            
            # Pose recovery
            _, R, t, pose_mask = cv2.recoverPose(
                E, curr_points, prev_points,
                focal=1.0, pp=(0, 0)
            )
            
            # Translációs vektor (up-to-scale)
            return t.flatten()
            
        except Exception as e:
            print(f"Essential matrix becslési hiba: {e}")
            return np.zeros(3)

    def estimate_scale_and_update_position(self, motion_vector): #Egy kamerából nem lehet megállapítani a valódi távolságot - ez a "skála probléma". Ha 1 métert vagy 10 métert mozdul a kamera, ugyanúgy néz ki.
    #A scale_factor=0.1 szorzót használ, amit az objektumok méretváltozása alapján finomít.
        """
        2. FÁZIS: Skála becslés és pozíció frissítés
        """
        if np.linalg.norm(motion_vector) < 1e-6:
            return np.zeros(3)
        
        # 2.A) IMU preintegráció (egyszerűsített - orientációból becsülve)
        with self.lock:
            # Orientáció alapú skála korrekció
            quat = [self.quaternion_w, self.quaternion_x, self.quaternion_y, self.quaternion_z]
            rotation_matrix = self.quaternion_to_rotation_matrix(*quat)
        
        # 2.B) Skála becslés (egyszerűsített)
        # Itt használhatnánk IMU gyorsulás adatokat ha elérhetőek
        estimated_scale = self.ar_tracker.scale_factor
        
        # Mozgásvektor skálázása és világkoordinátákba transzformálása
        world_motion = rotation_matrix @ (motion_vector * estimated_scale)
        
        # Pozíció frissítése
        self.ar_tracker.position_3d += world_motion
        
        # Sebesség becslés (egyszerűsített)
        self.ar_tracker.velocity_3d = world_motion * 30  # 30 FPS feltételezés
        
        return world_motion

    def update_bundle_adjustment(self, prev_points, curr_points, motion): #egyszerre optimalizálja a kamera pozíciót és a 3D pontokat, hogy minimalizálja a hibákat
        """
        3. FÁZIS: Bundle Adjustment (egyszerűsített sliding window)
        """
        # 3.A) Keyframe hozzáadása
        if len(self.ar_tracker.keyframes) == 0 or \
           np.linalg.norm(motion) > 0.1:  # Jelentős mozgás esetén új keyframe
            
            current_pose = {
                'id': self.ar_tracker.current_frame_id,
                'position': self.ar_tracker.position_3d.copy(),
                'motion': motion
            }
            
            self.ar_tracker.keyframes.append({
                'id': self.ar_tracker.current_frame_id,
                'points': curr_points.copy()
            })
            
            self.ar_tracker.frame_poses.append(current_pose)
        
        # 3.B) Sliding window optimalizáció (egyszerűsített)
        if len(self.ar_tracker.frame_poses) > 5:
            # Reprojection error minimalizáció (koncepcionális)
            self.optimize_sliding_window()

    def optimize_sliding_window(self):
        """
        3.A) Reprojection Error Minimalizáció (egyszerűsített)
        """
        # Itt lenne a teljes bundle adjustment implementáció
        # Egyszerűsített verzió: átlag pozíció smoothing
        if len(self.ar_tracker.frame_poses) >= 3: #Jelenleg csak az utolsó 3 pozíció átlagát veszi simítás céljából.
            recent_positions = [pose['position'] for pose in list(self.ar_tracker.frame_poses)[-3:]]
            smoothed_position = np.mean(recent_positions, axis=0)
            self.ar_tracker.position_3d = smoothed_position

    def detect_loop_closure(self, frame): #Megvizsgálja, hogy a robot visszatért-e egy korábban már látott helyre
        """
        4. FÁZIS: Loop Closure Detection (egyszerűsített)
        """
        # 4.A) BoW alapú (egyszerűsített verzió)
        # Itt ORB feature-ök hasonlítása korábbi frame-ekkel
        if self.ar_tracker.current_frame_id % 30 == 0:  # Minden 30. frame-nél
            # Egyszerűsített loop detection
            if len(self.ar_tracker.keyframes) > 10:
                # Itt lenne a bag-of-words összehasonlítás
                self.check_loop_closure()

    def check_loop_closure(self):
        """
        4.B) Pose Graph Optimization (egyszerűsített)
        """
        # Egyszerűsített loop closure korrekció
        # Drift csökkentése hosszú távon
        if len(self.ar_tracker.frame_poses) > 20:
            # Pozíció drift korrekció
            position_drift = np.linalg.norm(self.ar_tracker.position_3d)
            if position_drift > 10.0:  # 10 méter drift esetén reset
                print("Nagy drift észlelve - pozíció korrekció")
                self.ar_tracker.position_3d *= 0.8  # 20% korrekció

    def detect_and_track_objects(self, frame): #gyetlen menetben azonosítja és osztályozza az objektumokat. Egyedi ID-t ad minden objektumnak és követi a videó során. Ha egy objektum nagyobb lesz → közeledés, ha kisebb → távolodás, és ennek megfelelően állítja a skála faktort.
        """
        YOLO objektum detektálás és tracking
        """
        try:
            results = self.ar_tracker.yolo_model(frame)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.conf[0] > 0.5:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            area = (x2 - x1) * (y2 - y1)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Objektum követés logika
                            self.update_object_tracking(center_x, center_y, area)
                            
                            # Vizualizáció (opcionális)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            pass  # YOLO hibák elnyomása

    def update_object_tracking(self, center_x, center_y, area):
        """
        Objektum mozgás elemzés és pozíció finomhangolás
        """
        current_time = time.time()
        
        if 'main_object' not in self.ar_tracker.tracked_objects:
            self.ar_tracker.tracked_objects['main_object'] = {
                'center': (center_x, center_y),
                'area': area,
                'time': current_time,
                'history': deque(maxlen=10)
            }
        else:
            obj = self.ar_tracker.tracked_objects['main_object']
            prev_center = obj['center']
            prev_area = obj['area']
            
            # Mozgás irány becslés
            dx = center_x - prev_center[0]
            dy = center_y - prev_center[1]
            area_change = area - prev_area
            
            # Objektum alapú skála korrekció
            if abs(area_change) > 1000:
                if area_change > 0:
                    # Közeledés
                    self.ar_tracker.scale_factor = min(0.2, self.ar_tracker.scale_factor * 1.1)
                else:
                    # Távolodás
                    self.ar_tracker.scale_factor = max(0.05, self.ar_tracker.scale_factor * 0.9)
            
            # Frissítés
            obj['center'] = (center_x, center_y)
            obj['area'] = area
            obj['time'] = current_time
            obj['history'].append((dx, dy, area_change))

    def quaternion_to_rotation_matrix(self, w, x, y, z):
        """Kvaternió -> rotációs mátrix konverzió"""
        # Normalizálás
        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Koordináta rendszer korrekció
        x, y, z = -x, y, -z
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])

    def update_camera_orientation(self):
        """Kamera orientáció frissítés orientációból és pozícióból"""
        with self.lock:
            q_w, q_x, q_y, q_z = self.quaternion_w, self.quaternion_x, self.quaternion_y, self.quaternion_z
            
        # Kvaternió -> rotációs mátrix
        R = self.quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)
        # Kamera irányok frissítése
        self.camera_front = R[:, 2]  # Z tengely
        self.camera_up = R[:, 1]     # Y tengely
        
        # Normalizálás
        self.camera_front /= np.linalg.norm(self.camera_front)
        self.camera_up /= np.linalg.norm(self.camera_up)

    def process_input(self, delta_time):
        """Input kezelés (eredeti kód)"""
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
        
        # Manuális mozgás (felülírja az AR tracking-et)
        if np.linalg.norm(move_direction) > 0:
            move_direction /= np.linalg.norm(move_direction)
            self.camera_pos += move_direction * self.movement_speed * delta_time
        
        return True

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
        
        # Alpha shape rendering
        if self.mesh_triangles is not None and self.mesh_vertices is not None:
            glEnable(GL_BLEND) #színkeverés engedélyezése
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) #forrás szín alpha komponensével szoroz
            
            glBegin(GL_TRIANGLES)
            for tri in self.mesh_triangles:
                for idx in tri:
                    vertex = self.mesh_vertices[idx]
                    distance = np.linalg.norm(vertex - self.camera_pos)
                    t = min(distance / self.max_distance, 1.0)
                    
                    if t < 0.5:
                        r, g, b = t * 2, 1.0 - t * 2, 1.0
                    else:
                        t2 = (t - 0.5) * 2
                        r, g, b = 1.0, 1.0 - t2, 1.0 - t2
                    
                    glColor4f(r, g, b, 0.3)
                    glVertex3fv(vertex)
            glEnd()
            
            glDisable(GL_BLEND)
        
        pygame.display.flip()

    def run(self):
        """Fő futási loop"""
        clock = pygame.time.Clock()
        last_print_time = time.time()
        running = True
        
        while running:
            delta_time = clock.tick(60) / 1000.0
            running = self.process_input(delta_time)
            self.render()
            self.send_pose_data()

            
            # Debug információk
            if time.time() - last_print_time > 1.0:
                visible_points, _ = self.get_visible_points()
                print(f"Tracking State: {self.ar_tracker.tracking_state}")
                print(f"AR Position: {self.ar_tracker.position_3d}")
                print(f"Camera Position: {self.camera_pos}")
                print(f"Visible Points: {len(visible_points)}")
                print(f"Scale Factor: {self.ar_tracker.scale_factor:.3f}")
                print("-" * 50)
                last_print_time = time.time()
        
        # Cleanup
        self.running = False
        if hasattr(self, 'orientation_thread'):
            self.orientation_thread.join(timeout=1)
        if hasattr(self, 'vision_thread'):
            self.vision_thread.join(timeout=1)
        
        if self.cap:
            self.cap.release()
        self.socket.close()
        self.data_socket.close()

        pygame.quit()

    def send_pose_data(self):
        with self.data_socket_lock:
            pose_data = {
                "position": self.ar_tracker.position_3d.tolist(),
                "orientation": {
                    "w": self.quaternion_w,
                    "x": self.quaternion_x,
                    "y": self.quaternion_y,
                    "z": self.quaternion_z
                },
                "camera_front": self.camera_front.tolist()
            }
            try:
                json_str = json.dumps(pose_data) + '\n'
                self.data_socket.sendall(json_str.encode())
            except Exception as e:
                print(f"Data send error: {e}")


if __name__ == "__main__":
    try:
        viewer = PointCloudViewer(
            point_cloud_file="cave_sampled.ply", 
            host='127.0.0.1',  # Rust szerver
            port=12345,
            camera_id=0  # USB kamera
        )
        viewer.run()
    except Exception as e:
        print(f"Hiba: {e}")
        import traceback
        traceback.print_exc()
