import cv2
import numpy as np
from ultralytics import YOLO

# 🔧 Пути
image_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/data/barier/images/test/67.jpg'
label_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/data/barier/labels/test/67.txt'
model_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/data/barier/runs/detect/train/weights/best.pt'

# 🎛️ Конфигурации порогов
param_sets = [
    (0.25, 0.5),
    (0.6, 0.7),
    (0.4, 0.3)
]

distance_thresh = 50  # допустимое расстояние центра

# 📐 Метрики
def iou(boxA, boxB):
    xa, ya, xa2, ya2 = boxA
    xb, yb, xb2, yb2 = boxB
    inter_x1 = max(xa, xb)
    inter_y1 = max(ya, yb)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = max(1, (xa2 - xa) * (ya2 - ya))
    areaB = max(1, (xb2 - xb) * (yb2 - yb))
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0

def center(box):
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return x, y

def center_distance(boxA, boxB):
    ax, ay = center(boxA)
    bx, by = center(boxB)
    return ((ax - bx)**2 + (ay - by)**2)**0.5

# 🧠 Загрузка модели
model = YOLO(model_path)

# 📷 Загрузка изображения и аннотаций
img = cv2.imread(image_path)
h, w = img.shape[:2]

gt_boxes = []
with open(label_path, 'r') as f:
    for line in f:
        cls, cx, cy, bw, bh = map(float, line.strip().split())
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        gt_boxes.append([x1, y1, x2, y2])

# 🔁 Проход по параметрам
for conf_thresh, iou_thresh in param_sets:
    print(f'\n==============================')
    print(f'🔧 Параметры: conf = {conf_thresh}, iou = {iou_thresh}')
    print(f'📌 Распознавание по IoU и центру:')

    results = model.predict(img, conf=conf_thresh, verbose=False)[0]
    pred_boxes = [list(map(int, box.xyxy[0])) for box in results.boxes if float(box.conf[0]) >= conf_thresh]

    iou_values = []
    center_distances = []

    for idx, pred in enumerate(pred_boxes):
        best_iou = 0
        best_dist = None
        for gt in gt_boxes:
            current_iou = iou(pred, gt)
            dist = center_distance(pred, gt)
            if current_iou > best_iou:
                best_iou = current_iou
                best_dist = dist
        iou_values.append(best_iou)
        center_distances.append(best_dist)
        if best_dist is not None:
            match_status = best_iou >= iou_thresh and best_dist <= distance_thresh
            print(f'Prediction #{idx+1} → IoU: {best_iou:.2f}, CenterDist: {best_dist:.1f}px — '
                  + ('MATCH ✅' if match_status else 'No match ❌'))
        else:
            print(f'Prediction #{idx+1} → IoU: {best_iou:.2f}, CenterDist: N/A — No match ❌')

    # 📈 Средние значения (с фильтрацией None)
    valid_dists = [d for d in center_distances if d is not None]
    mean_iou = sum(iou_values) / len(iou_values) if iou_values else 0
    mean_dist = sum(valid_dists) / len(valid_dists) if valid_dists else 0

    print(f'\n📐 Средний IoU по предсказаниям: {mean_iou:.3f}')
    print(f'📏 Среднее расстояние между центрами: {mean_dist:.2f}px')
