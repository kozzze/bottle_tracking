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

'''fourcc = cv.VideoWriter_fourcc(*'avc1')
cap_write = cv.VideoWriter(path_save, fourcc, fps, (width, height))
while True:
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

