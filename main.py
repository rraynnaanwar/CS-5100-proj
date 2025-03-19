import cv2
import torch
from ultralytics import YOLO

model = YOLO('models/players/best.pt')
print(model.names)
vid = cv2.VideoCapture('vids/test vid.mp4')
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
            class_id = int(box.cls[0])  # Class ID (person=0 for COCO)
            if class_id == 0:
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)  # Take the bottom center point (feet)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            elif class_id ==2:
                center_x = int((x1 + x2) / 2)
                center_y = int(y1+y2)  # Take the bottom center point (feet)
                radius = int(max((x2-x1) // 2, (y2-y1) // 2)) 
                cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 3)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
vid.release()
cv2.destroyAllWindows()