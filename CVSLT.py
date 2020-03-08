import cv2
import numpy as np
import math
from scipy.signal import find_peaks

def nothing(x):
    pass

# def checkFingers(frame):
#

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
edges = cv2.namedWindow('edges')
lower = np.array([0, 40, 0], dtype = "uint8")
upper = np.array([15, 255, 255], dtype = "uint8")
backSub = cv2.createBackgroundSubtractorKNN()

while(ret):
    frame = cv2.flip(frame, 1)
    box = frame[50:410, 760:1080, :]
    fgMask = backSub.apply(box)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fgMask = cv2.erode(fgMask, kernel, iterations = 2)
    fgMask = cv2.dilate(fgMask, kernel, iterations = 2)
    fgMask = cv2.GaussianBlur(fgMask, (17, 17), 0)
    _, contours, _ = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(contours):
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        cv2.drawContours(box, contours, -1, (0, 0, 255), 2)
        if (cv2.contourArea(cnt) > 15000):
        # if (cv2.contourArea(cnt) < 1000):
            cnt = np.asarray(cnt)
            M = cv2.moments(cnt)
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            validPoints = []
            for pt in cnt:
                if(pt[0][1] > y):
                    validPoints.append(pt)

            validPoints = np.asarray(validPoints)
            dist = np.empty(validPoints.shape[0])
            print(validPoints.shape)
            for point in validPoints:
                # dist = np.append(dist, math.sqrt(((point[0][0]-x)**2)+(point[0][1]-y)**2))
                dist = np.append(dist, math.sqrt(((point[0][1]-y)**2)))

            maxima, _ = find_peaks(dist)
            print(maxima.shape)
            # for ind in maxima:
            #     cv2.circle(box, (validPoints[ind][0][0], validPoints[ind][0][1]), 5, (255, 0, 0))
            # if(maxima.shape[0] == 5):
            #     for ind in maxima:
            #         cv2.circle(box, validPoints[0][0][ind], 5, (255, 0, 0))

    skin = cv2.bitwise_and(box, box, mask = fgMask)
    cv2.rectangle(frame, (760, 50), (1080, 410), (0, 255, 0), 2)
    cv2.imshow('original', frame)
    # cv2.imshow('test', box)
    if(cv2.waitKey(1) == ord('q')):
        break
    ret, frame = cap.read()
