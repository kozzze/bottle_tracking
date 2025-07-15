import cv2

# 📷 Загрузка изображения
image_path = '/Users/kozzze/Desktop/home.png'  # путь к фото
img = cv2.imread(image_path)
roi = cv2.selectROI("Выбери Зону Интереса (ROI)", img, showCrosshair=True, fromCenter=False)

# 🖼️ Отрисовка прямоугольника
(x, y, w, h) = roi
roi_box = (x, y, x + w, y + h)
cv2.rectangle(img, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 255, 0), 2)
cv2.imshow("ROI выделено", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 💾 Сохраняем координаты
print(f"Зона интереса: x_min={roi_box[0]}, y_min={roi_box[1]}, x_max={roi_box[2]}, y_max={roi_box[3]}")
