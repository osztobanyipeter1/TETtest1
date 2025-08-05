import cv2
import numpy as np
import itertools
import time
from ultralytics import YOLO

# YOLO előkészítés
model = YOLO('yolov8s.pt')

# Optical flow paraméterek
lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners = 20,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )

trajectory_len = 20
detect_interval = 1
trajectories = []
frame_idx = 0
prev_gray = None

cap = cv2.VideoCapture(0)

# Állapotváltozók
prev_area = None
prev_avg_dist = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()
    direction_texts = []

    # --- OBJECT DETECTION (YOLO) ---
    results = model(frame)
    largest_area = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    if largest_area > 0:
        if prev_area is not None:
            if largest_area > prev_area + 1000:
                dir_obj = "közeledik"
            elif largest_area < prev_area - 1000:
                dir_obj = "távolodik"
            else:
                dir_obj = "Nincs változás"
            direction_texts.append(f"Obj: {dir_obj}")
        prev_area = largest_area

    # --- CORNER TRACKING (OPTICAL FLOW) ---
    if len(trajectories) > 0 and prev_gray is not None:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([traj[-1] for traj in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_traj = []
        for traj, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            traj.append((x, y))
            if len(traj) > trajectory_len:
                del traj[0]
            new_traj.append(traj)
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
        trajectories = new_traj
        cv2.polylines(img, [np.int32(traj) for traj in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(traj[-1]) for traj in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])
    if len(trajectories) > 1:
        points = np.array([traj[-1] for traj in trajectories])
        dists = []
        for (x1, y1), (x2, y2) in itertools.combinations(points, 2):
            dists.append(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        avg_dist = np.mean(dists)
        if prev_avg_dist is not None:
            if avg_dist > prev_avg_dist + 1:
                dir_corn = "távolodik"
            elif avg_dist < prev_avg_dist - 1:
                dir_corn = "közeledik"
            else:
                dir_corn = "Nincs változás"
            direction_texts.append(f"Sarok: {dir_corn}")
        prev_avg_dist = avg_dist

    # --- VÉGSŐ DÖNTÉS ---
    # Csak akkor adj értelmes választ, ha mindkét mérés van!
    if len(direction_texts) == 2:
        if "közeledik" in direction_texts[0] and "közeledik" in direction_texts[1]:
            final_direction = "Végső: közeledik"
        elif "távolodik" in direction_texts[0] and "távolodik" in direction_texts[1]:
            final_direction = "Végső: távolodik"
        else:
            final_direction = "Végső: nem egyértelmű"
        direction_texts.append(final_direction)

    y0 = 80
    for i, txt in enumerate(direction_texts):
        cv2.putText(img, txt, (30, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 200), 3)

    cv2.imshow("YOLO + OpticalFlow", img)
    frame_idx += 1
    prev_gray = frame_gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
