import os
import cv2

# ⚙️ Папки с изображениями и разметкой
img_dir = 'obj_train_data'
label_dir = 'obj_train_data'
output_dir = 'preview_annotated'
os.makedirs(output_dir, exist_ok=True)

# Размеры (нужны для рескейла координат)
def draw_boxes(image_path, label_path, save_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # пропустить кривую строку
            class_id, x, y, w, h = map(float, parts)

            # Преобразуем YOLO → pixel координаты
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            # Рисуем прямоугольник
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Bottle", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(save_path, img)

# 🌀 Обработка всех пар
for filename in os.listdir(img_dir):
    if filename.endswith('.png'):
        base = os.path.splitext(filename)[0]
        img_path = os.path.join(img_dir, filename)
        label_path = os.path.join(label_dir, base + '.txt')
        out_path = os.path.join(output_dir, filename)

        if os.path.exists(label_path):
            draw_boxes(img_path, label_path, out_path)

print(f"[✓] Все изображения размечены и сохранены в папке {output_dir}")
