import time
import cv2

width = 1280
height = 720

cv2.namedWindow('left', cv2.WINDOW_NORMAL)
cv2.namedWindow('right', cv2.WINDOW_NORMAL)
camera = cv2.VideoCapture(2)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

counter = 0
utc = time.time()
folder = './photo/'



def shot(pos, frame):
    global counter
    path = folder + pos + '/' + pos + str(counter) + ".jpg"
    cv2.imwrite(path, frame)
    print('snapshot save into: ' + path)


while True:
    ret, frame = camera.read()

    left_frame = frame[0:height, 0:width]
    right_frame = frame[0:height, width:width*2]

    cv2.imshow('left', left_frame)
    cv2.imshow('right', right_frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1

camera.release()
cv2.destroyWindow('left')
cv2.destroyWindow('right')