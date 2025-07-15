import cv2
from ultralytics import YOLO

video_path = '/Users/kozzze/Desktop/home3.mp4'
model_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/data/bottle_front_data/runs/detect/train3/weights/best.pt'

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1280, 720))

    results = model.predict(source=frame, conf=0.25, imgsz=1280, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Bottle {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Video Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
