import cv2
import os

def extract_frames(video_path, output_folder, num_frames, start_index):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return start_index

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames, 1)

    extracted_count = 0
    frame_id = 0

    while extracted_count < num_frames and frame_id < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        filename = os.path.join(output_folder, f"{start_index}.jpg")
        cv2.imwrite(filename, frame)

        extracted_count += 1
        frame_id += interval
        start_index += 1

    cap.release()
    print(f"{video_path}: извлечено {extracted_count} кадров.")
    return start_index

# Список видеофайлов
video_paths = [
    '/Users/kozzze/Desktop/1.MOV',
    '/Users/kozzze/Desktop/2.MOV',
    '/Users/kozzze/Desktop/3.MOV',
    '/Users/kozzze/Desktop/4.MOV',
    '/Users/kozzze/Desktop/5.MOV'
]

# Общее количество кадров
total_desired_frames = 75

# Папка для всех кадров
common_output_folder = "all_frames"

# Сколько кадров извлекать с каждого видео (равномерно)
frames_per_video = total_desired_frames // len(video_paths)

# Начальная нумерация
current_index = 1

# Извлечение кадров
for video_path in video_paths:
    current_index = extract_frames(video_path, common_output_folder, frames_per_video, current_index)
