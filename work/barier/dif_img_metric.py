import cv2
import numpy as np
from ultralytics import YOLO

# 🔧 Параметры
image_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/work/barier/images/test/74.jpg'
label_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/work/barier/labels/test/74.txt'
model_path = '/Users/kozzze/Desktop/Учеба/bottle_tracking/work/barier/runs/detect/train/weights/best.pt'
conf_thresh = 0.3
iou_thresh = 0.3
distance_thresh = 50

# 📦 Модель
model = YOLO(model_path)

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

# 🧪 Предобработки
def enhance_image(img):
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def contour_enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = np.zeros_like(img)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return cv2.addWeighted(img, 0.8, overlay, 0.2, 0)

# 📊 Анализ
def analyze_image(img, gt_boxes, model):
    annotated = img.copy()
    results = model.predict(img, conf=conf_thresh, verbose=False)[0]
    preds = [list(map(int, box.xyxy[0])) for box in results.boxes if float(box.conf[0]) >= conf_thresh]

    matched_gt = set()
    matched_pred = set()
    iou_list = []
    dist_list = []
    tp, fp = 0, 0

    for i, pred in enumerate(preds):
        best_iou, best_gt = 0, None
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou_score = iou(pred, gt)
            dist = center_distance(pred, gt)
            if iou_score >= iou_thresh and dist <= distance_thresh and iou_score > best_iou:
                best_iou, best_gt = iou_score, j
        if best_gt is not None:
            matched_gt.add(best_gt)
            matched_pred.add(i)
            tp += 1
            iou_list.append(best_iou)
            dist_list.append(center_distance(pred, gt_boxes[best_gt]))
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
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    mean_dist = sum(dist_list) / len(dist_list) if dist_list else 0
    map_at_iou = tp / len(preds) if preds else 0

    return {
        'image': annotated,
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'mean_iou': mean_iou,
        'mean_dist': mean_dist,
        'map': map_at_iou,
        'gt_count': len(gt_boxes),
        'pred_count': len(preds)
    }

# 📥 Загрузка
img_orig = cv2.imread(image_path)
img_enh = enhance_image(img_orig.copy())
img_contour = contour_enhance(img_orig.copy())
h, w = img_orig.shape[:2]

# 📂 GT боксы
gt_boxes = []
with open(label_path, 'r') as f:
    for line in f:
        cls, cx, cy, bw, bh = map(float, line.strip().split())
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        gt_boxes.append([x1, y1, x2, y2])

# 📊 Анализ
res_orig = analyze_image(img_orig, gt_boxes, model)
res_enh = analyze_image(img_enh, gt_boxes, model)
res_contour = analyze_image(img_contour, gt_boxes, model)

# 📋 Метрики
def print_metrics(name, res):
    print(f"\n📊 {name}")
    print(f"GT-боксов:       {res['gt_count']}")
    print(f"Предсказано:     {res['pred_count']}")
    print(f"✅ TP: {res['tp']}   ❌ FP: {res['fp']}   ⚠️ FN: {res['fn']}")
    print(f"📐 Средний IoU:      {res['mean_iou']:.3f}")
    print(f"📏 Ср. центр-расст.: {res['mean_dist']:.2f}px")
    print(f"🎯 Precision:        {res['precision']:.3f}")
    print(f"📈 Recall:           {res['recall']:.3f}")
    print(f"🔁 F1-score:         {res['f1']:.3f}")
    print(f"📦 mAP@IoU:          {res['map']:.3f}")

print_metrics("ОРИГИНАЛ", res_orig)
print_metrics("УЛУЧШЕННОЕ (GRAY+αβ)", res_enh)
print_metrics("КОНТУРНОЕ", res_contour)

def resize_for_display(img, scale=0.4):
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

# 🔄 Уменьшаем изображения для отображения
disp_orig = resize_for_display(res_orig['image'])
disp_enh = resize_for_display(res_enh['image'])
disp_contour = resize_for_display(res_contour['image'])

# 🖼️ Объединяем горизонтально
comparison = cv2.hconcat([disp_orig, disp_enh, disp_contour])
cv2.imshow("Original | Enhanced | Contour", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()

