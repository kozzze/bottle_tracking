import cv2
from ultralytics import YOLO
import pandas as pd

video_path = '/Users/kozzze/Desktop/IMG_6355.MOV'
model_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/data/barier/runs/detect/train/weights/best.pt'
output_csv = 'bottle3_log.csv'
save_video = True

roi = (500, 500, 2000, 700)

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_with_boxes.mp4', fourcc, fps, (w, h))

counted_ids = set()
log = []

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1

    results = model.track(source=frame, persist=True, verbose=False, imgsz=1280)

    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)

    for result in results:
        for box in result.boxes:
            if box.id is None:
                continue
            id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                if id not in counted_ids:
                    counted_ids.add(id)
                    timestamp = frame_count / fps
                    log.append({'Bottle_ID': id, 'Frame': frame_count, 'Time_sec': round(timestamp, 2)})
                    print(f'[+] Бутылка {id} вошла в ROI в кадре {frame_count}')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if save_video:
        out.write(frame)

    cv2.imshow('Bottle Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if save_video:
    out.release()

df = pd.DataFrame(log)
df.to_csv(output_csv, index=False)
print