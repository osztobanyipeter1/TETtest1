import cv2
import numpy as np
from engine.object_detection import ObjectDetection #az fix, hogy nekem nincsen ilyen engine.object_detection fileom, szoval ehelyett kell valami
from engine.object_tracking import MultiObjectTracking

#load the object detection model
od = ObjectDetection("models/yolo11m.pt") #https://cocodataset.org/#explore

#load video
cap = cv2.VideoCapture('video.mp4')

#load the object tracking model
mot = MultiObjectTracking()
tracker = mot.ocsort(min_hits=3, max_age=10, iou_threshold=0.5) #nemtudom mik ezek de lehet nem is kell

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #detect objects in the frame
    bboxes, class_ids, scores = od.detect(frame)
    #for bbox, class_id, score in zip(bboxes, class_ids, scores):
    #    x1, y1, x2, y2 =bbox
    #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #    cv2.putText(frame, od.classes[class_id], (x1,y1-10),
    #                cv2.FONT_HERSEY_SIMPLEX, 0.9, (36,255,12), 2)

    # update the tracker
    bboxes_ids = tracker.update(bboxes, class_ids, scores, frame)
    for bbox_id in bboxes_ids:
        (x1, y1, x2, y2, obj_id, class_id, score) = np.array(bbox_id)
        cv2.rectangle(frame, (x1,y1), (x2, y2), (255, 0, 0), 2)

        #Display object ID
        cv2.putText(frame, f"ID:{obj_id}", (x1, y1-5),
                    cv2.FONT_HERSEY_SIMPLEX, 1, (255, 0, 0), 2)


    cv2.imshow(frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()

# https://www.youtube.com/watch?v=w-IuLVibtWM