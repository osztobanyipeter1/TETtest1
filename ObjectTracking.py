import cv2
from ultralytics import YOLO

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture(4)

prev_area = None
direction_text = ""
prev_centre_x = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    largest_area = 0
    largest_center_x = None

    for result in results:
        boxes = result.boxes
        #classes_names = result.names
        for box in boxes:
            if box.conf[0] > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_center_x = (x1 + x2) // 2
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    area_movement = ""
    lateral_movement = ""

    if largest_area > 0:
        if prev_area is not None:
            if largest_area > prev_area + 1000:
                direction_text = "közeledik"
            elif largest_area < prev_area - 1000:
                direction_text = "távolodik"
            else:
                direction_text = "Nincs változás"
        prev_area = largest_area

    if largest_center_x is not None:
        if prev_centre_x is not None:
            if largest_center_x > prev_centre_x + 10:
                lateral_movement = "jobbra mozog"
            elif largest_center_x < prev_centre_x - 10:
                lateral_movement = "balra mozog"
            else:
                lateral_movement = " nincs oldalra mozgás"
        prev_centre_x = largest_center_x

    info = ""
    if area_movement:
        info += area_movement
    if lateral_movement:
        info += (", " if info else "") + lateral_movement

    #cv2.putText(frame, direction_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    print(info)

    cv2.imshow("YOLO Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
