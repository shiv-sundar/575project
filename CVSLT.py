import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
backSub = cv2.createBackgroundSubtractorKNN()
while(ret):
    frame = cv2.flip(frame, 1)
    box = frame[50:450, 760:1110, :]
    t = box.copy()
    t = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    fgMask = backSub.apply(t)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fgMask = cv2.GaussianBlur(fgMask, (5, 5), 0)
    fgMask = cv2.erode(fgMask, kernel, iterations = 3)
    hand = cv2.dilate(fgMask, kernel, iterations = 3)
    _, contours, _ = cv2.findContours(hand.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for x in range(1): #I'm sorry this is bad but I just need to break out of a specific point still in the loop
        if(contours):
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            area = cv2.contourArea(cnt)
            if((area > 20000) & (area < 50000)):
                M = cv2.moments(cnt)
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                cv2.circle(box, (x, y), 3, (0, 0, 255), -1)
                if ((x < 150) | (x > 200) | (y < 225) | (y > 275)):
                    cv2.putText(frame, "Try to match the red dot to the blue dot", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness = 2)
                    continue

                deg = np.zeros(0)
                dist = np.zeros(0)
                for point in cnt:
                    angle = math.degrees(math.atan2(point[0][1] - y, point[0][0] - x))
                    deg = np.append(deg, angle)
                    distance = math.sqrt(((point[0][0] - x)**2) + (point[0][1] - y)**2)
                    dist = np.append(dist, distance)

                # maxima, _ = find_peaks()

            elif(area >= 50000):
                cv2.putText(frame, "Hand is too close", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness = 2)

            else:
                cv2.putText(frame, "Hand is too far", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness = 2)

    skin = cv2.bitwise_and(box, box, mask = fgMask)
    cv2.circle(box, (175, 250), 3, (255, 0, 0), -1)
    cv2.rectangle(frame, (760, 50), (1110, 450), (0, 255, 0), 1)
    cv2.imshow('original', frame)
    if(cv2.waitKey(1) == ord('q')):
        cap.release()
        break
    ret, frame = cap.read()
