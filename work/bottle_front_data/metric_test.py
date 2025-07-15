import cv2
import numpy as np
from ultralytics import YOLO

# 🔧 Задать параметры
image_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/data/bottle_front_data/images/test/2.jpg'
label_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/data/bottle_front_data/labels/test/2.txt'
model_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/data/bottle_front_data/runs/detect/train3/weights/best.pt'
conf_thresh = 0.5
iou_thresh = 0.25
distance_thresh = 50  # допустимое расстояние между центрами

# 📦 Загрузка модели
model = YOLO(model_path)

# 📷 Загрузка изображения
img = cv2.imread(image_path)
h, w = img.shape[:2]

# 🟥 Загрузка разметки
gt_boxes = []
with open(label_path, 'r') as f:
    for line in f:
        cls, cx, cy, bw, bh = map(float, line.strip().split())
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        gt_boxes.append([x1, y1, x2, y2])

# 🟩 Предсказания модели
results = model.predict(img, conf=conf_thresh, verbose=False)[0]
pred_boxes = [list(map(int, box.xyxy[0])) for box in results.boxes if float(box.conf[0]) >= conf_thresh]

# 📐 Функция IoU
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

# 📍 Центры и расстояние между ними
def center(box):
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return x, y

def center_distance(boxA, boxB):
    ax, ay = center(boxA)
    bx, by = center(boxB)
    return ((ax - bx)**2 + (ay - by)**2)**0.5

# 📊 Анализ предсказаний
iou_values = []

print('\n📌 Распознавание по IoU и центру:')
for idx, pred in enumerate(pred_boxes):
    best_iou = 0
    best_dist = None
    best_gt = None
    for gt in gt_boxes:
        current_iou = iou(pred, gt)
        dist = center_distance(pred, gt)
        if current_iou > best_iou:
            best_iou = current_iou
            best_dist = dist
            best_gt = gt
    iou_values.append(best_iou)
    match_status = best_iou >= iou_thresh and best_dist <= distance_thresh
    color = (0, 255, 0) if match_status else (255, 255, 0)
    cv2.rectangle(img, (pred[0], pred[1]), (pred[2], pred[3]), color, 2)
    print(f'Prediction #{idx+1} → IoU: {best_iou:.2f}, CenterDist: {best_dist:.1f}px — '
          + ('MATCH ✅' if match_status else 'No match ❌'))

# 🟥 Ground Truth — красным
for gt in gt_boxes:
    cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0, 0, 255), 2)

# 📈 Средний IoU
if iou_values:
    mean_iou = sum(iou_values) / len(iou_values)
    print(f'\n📐 Средний IoU по предсказаниям: {mean_iou:.3f}')
else:
    print('\n❗ Нет предсказаний для расчёта среднего IoU')
# 📊 Сбор расстояний между центрами
center_distances = []

for pred in pred_boxes:
    min_dist = None
    for gt in gt_boxes:
        dist = center_distance(pred, gt)
        if (min_dist is None) or (dist < min_dist):
            min_dist = dist
    if min_dist is not None:
        center_distances.append(min_dist)

# 📈 Среднее расстояние между центрами
if center_distances:
    mean_center_dist = sum(center_distances) / len(center_distances)
    print(f'📏 Среднее расстояние между центрами: {mean_center_dist:.2f}px')
else:
    print('❗ Нет данных для расчёта среднего расстояния')
# 🖼️ Отображение
cv2.imshow('Ground Truth (Red) vs Predictions (Green/Yellow)', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
