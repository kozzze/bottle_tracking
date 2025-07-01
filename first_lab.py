import cv2 as cv

img = r'/Users/kozzze/Desktop/Учеба/bottle_tracking/img/4k.jpg'

#img_color = cv.imread(img, cv.IMREAD_COLOR)
#img_grey = cv.imread(img, cv.IMREAD_GRAYSCALE)
#img_unchanged = cv.imread(img, cv.IMREAD_UNCHANGED)

#cv.namedWindow('color',cv.WINDOW_NORMAL)
#cv.imshow('color', img_color)

#cv.namedWindow('grey',cv.WINDOW_AUTOSIZE)
#cv.imshow('grey', img_grey)

#cv.namedWindow('unchanged',cv.WINDOW_FREERATIO)
#cv.imshow('unchanged', img_unchanged)

#cv.waitKey(0)
#cv.destroyAllWindows()

#----------
cap = cv.VideoCapture(r'/Users/kozzze/Desktop/Учеба/bottle_tracking/video/conveier_front.mp4')

path_save = '/Users/kozzze/Desktop/Учеба/bottle_tracking/video/save_video.mp4'

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'avc1')
cap_write = cv.VideoWriter(path_save, fourcc, fps, (width, height))
'''while True:
    ret, frame = cap.read()
    if not ret:
        break
    cap_write.write(frame)
    if cv.waitKey(25) & 0xFF == 27:
        break
cap.release()
cap_write.release()
cv.destroyAllWindows()

img = cv.imread(img, cv.IMREAD_COLOR)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow('color', img)
cv.imshow('hsv', img_hsv)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#--------------------

'''video = cv.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    cv.line(frame,
             (width // 2, height // 2 - 50),
             (width // 2, height // 2 + 50),
             (0, 0, 255), 5)

    cv.line(frame,
             (width // 2 - 50, height // 2),
             (width // 2 + 50, height // 2),
             (0, 0, 255), 5)
    cv.imshow('video', frame)
    cap_write.write(frame)
    if cv.waitKey(25) & 0xFF == 27:
        break



cv.waitKey(0)
cv.destroyAllWindows()
'''
#__________-_

import numpy as np

def main_color(pixel):
    b,r,g = pixel
    distances = {
        'red': np.linalg.norm([r - 255, g, b]),
        'green': np.linalg.norm([r, g - 255, b]),
        'blue': np.linalg.norm([r, g, b - 255])
    }
    return min(distances, key=distances.get)

camera = cv.VideoCapture(0)
while True:
    ret, frame = camera.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    center_pixel = frame[center_y, center_x]
    color_name = main_color(center_pixel)

    if color_name == 'red':
        color = (0, 0, 255)
    elif color_name == 'green':
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    cv.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), color, 5)
    cv.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), color, 5)

    cv.imshow('Colored Cross', frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

