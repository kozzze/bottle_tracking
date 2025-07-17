import cv2
import numpy as np
from ultralytics import YOLO

# 🎨 Улучшенная обработка: grayscale + alpha/beta + контуры
def full_enhance(img):
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(gray_bgr)
    cv2.drawContours(contour_mask, contours, -1, (255,255,255), 1)
    return cv2.addWeighted(gray_bgr, 0.8, contour_mask, 0.2, 0)

# 📐 IoU и расстояние
def iou(boxA, boxB):
    xa, ya, xa2, ya2 = boxA
    xb, yb, xb2, yb2 = boxB
    inter_x1, inter_y1 = max(xa, xb), max(ya, yb)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = max(1, (xa2 - xa) * (ya2 - ya))
    areaB = max(1, (xb2 - xb) * (yb2 - yb))
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0

def center_distance(boxA, boxB):
    ax = (boxA[0] + boxA[2]) / 2
    ay = (boxA[1] + boxA[3]) / 2
    bx = (boxB[0] + boxB[2]) / 2
    by = (boxB[1] + boxB[3]) / 2
    return ((ax - bx)**2 + (ay - by)**2)**0.5

# 📊 Анализ предсказаний
def analyze(img, gt_boxes, model, conf_thresh, iou_thresh, dist_thresh):
    annotated = img.copy()
    results = model.predict(img, conf=conf_thresh, verbose=False)[0]
    preds = [list(map(int, box.xyxy[0])) for box in results.boxes if float(box.conf[0]) >= conf_thresh]

    matched_gt, matched_pred = set(), set()
    tp, fp = 0, 0
    ious, dists = [], []

    for i, pred in enumerate(preds):
        best_iou, best_gt = 0, None
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou_score = iou(pred, gt)
            dist = center_distance(pred, gt)
            if iou_score >= iou_thresh and dist <= dist_thresh and iou_score > best_iou:
                best_iou, best_gt = iou_score, j
        if best_gt is not None:
            matched_gt.add(best_gt)
            matched_pred.add(i)
            tp += 1
            ious.append(best_iou)
            dists.append(center_distance(pred, gt_boxes[best_gt]))
            cv2.rectangle(annotated, (pred[0], pred[1]), (pred[2], pred[3]), (0,255,0), 2)
        else:
            fp += 1
            cv2.rectangle(annotated, (pred[0], pred[1]), (pred[2], pred[3]), (255,0,255), 2)
    fn = len(gt_boxes) - tp
    for gt in gt_boxes:
        cv2.rectangle(annotated, (gt[0], gt[1]), (gt[2], gt[3]), (0,0,255), 2)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    mean_iou = sum(ious)/len(ious) if ious else 0
    mean_dist = sum(dists)/len(dists) if dists else 0
    map_at_iou = tp / len(preds) if preds else 0

    return annotated, {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'mean_iou': mean_iou,
        'mean_dist': mean_dist,
        'map': map_at_iou,
        'gt_count': len(gt_boxes),
        'pred_count': len(preds)
    }

# 🔧 Параметры
image_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/work/barier/images/test/74.jpg'
label_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/work/barier/labels/test/74.txt'
model_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/work/barier/runs/detect/train/weights/best.pt'
conf_thresh = 0.3
iou_thresh = 0.3
distance_thresh = 50
model = YOLO(model_path)

# 📥 Загрузка изображения и GT
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

# 🧪 Анализ: оригинал и улучшенное
img_enhanced = full_enhance(img.copy())
res1_img, res1_metrics = analyze(img, gt_boxes, model, conf_thresh, iou_thresh, distance_thresh)
res2_img, res2_metrics = analyze(img_enhanced, gt_boxes, model, conf_thresh, iou_thresh, distance_thresh)

# 📋 Вывод в консоль
def print_metrics(name, res):
    print(f"\n📊 {name}")
    print(f"GT-боксов: {res['gt_count']} | Предсказано: {res['pred_count']}")
    print(f"✅ TP: {res['tp']} | ❌ FP: {res['fp']} | ⚠️ FN: {res['fn']}")
    print(f"📐 Ср. IoU:       {res['mean_iou']:.3f}")
    print(f"📏 Ср. расст. ц.: {res['mean_dist']:.2f}px")
    print(f"🎯 Precision:     {res['precision']:.3f}")
    print(f"📈 Recall:        {res['recall']:.3f}")
    print(f"🔁 F1-score:      {res['f1']:.3f}")
    print(f"📦 mAP@IoU:       {res['map']:.3f}")

print_metrics("СТОКОВОЕ", res1_metrics)
print_metrics("ИЗМЕНЁННОЕ (GREY + αβ + Контуры)", res2_metrics)

# 🖼️ Объединённое окно
def resize(img, scale=0.4):
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

combined = cv2.hconcat([resize(res1_img), resize(res2_img)])
cv2.imshow("Left: Stock | Right: Enhanced", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
