import cv2
import numpy as np
import os
from pathlib import Path


def plot_yolo_bbox(image_path, txt_path):
    """Визуализирует YOLO-разметку на изображении с проверкой ошибок."""
    # Проверка существования файлов
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл изображения не найден: {image_path}")
        return
    if not os.path.exists(txt_path):
        print(f"Ошибка: Файл разметки не найден: {txt_path}")
        return

    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Изображение: {image_path} ({w}x{h})")

    # Чтение разметки
    with open(txt_path) as f:
        lines = f.readlines()

    # Отрисовка bbox
    for line in lines:
        try:
            cls, xc, yc, bw, bh = map(float, line.strip().split())
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Class {int(cls)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            print(f"Ббокс: класс {int(cls)}, центр ({xc:.2f},{yc:.2f}), размер {bw:.2f}x{bh:.2f}")
        except ValueError as e:
            print(f"Ошибка в строке разметки: {line.strip()} ({e})")

    # Показать результат
    cv2.imshow("YOLO BBox Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Пример использования (замените пути на свои)
image_path = "/Users/kozzze/Desktop/Учеба/bottle_tracking/data/img2:video/Снимок экрана 2025-07-03 в 11.30.29 AM.png"  # Полный путь к изображению
txt_path = "/Users/kozzze/Desktop/Учеба/bottle_tracking/data/bottle_front_data/labels/Снимок экрана 2025-07-03 в 11.30.29 AM.txt"  # Полный путь к разметке

plot_yolo_bbox(image_path, txt_path)