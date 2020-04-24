import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, peak_prominences, find_peaks

def checkHandClosed(cntData, centerX, centerY):

    """Use distance from center to top of contour,
    if relatively small, then hand closed"""

def countFingers(cntData, fingerLocation, centerX, centerY):
    #work on returning array like [[finger1, closed], [finger2, open], [finger3, open]]
    deg = np.zeros(0)
    dist = np.zeros(0)
    if(fingerLocation == "t"):
        for point in cntData:
            angle = math.degrees(math.atan2(point[0][1] - y, point[0][0] - x))
            deg = np.append(deg, angle)
            if (point[0][1] > centerY + 30):
                dist = np.append(dist, 0)
                continue
            distance = math.sqrt(((point[0][0] - x)**2) + (point[0][1] - y)**2)
            dist = np.append(dist, distance)

        #open
        maxima, dict = find_peaks(dist, height=20, distance=5, prominence=35)
        # maxima = argrelmax(dist, order = 25)
        # prominence = peak_prominences(dist, maxima[0])
        fingers = []

        # for finger in range(len(maxima[0])):
        #     if(prominence[0][finger] < 35):
        #         type = "closed"
        #
        #     else:
        #         type = "open"
        #
        #     fingers.append([finger, type])

        if (cv2.waitKey(1) == ord("w")):
            print(maxima)
        # return fingers

    elif(fingerLocation == "i"):
        """Todo: need to do internal contour detection"""
        # for point in cntData:
        #     if (point[0][1] > 330):
        #         dist = np.append(dist, 0)
        #         continue
        #     angle = math.degrees(math.atan2(point[0][1] - y, point[0][0] - x))
        #     deg = np.append(deg, angle)
        #     distance = math.sqrt(((point[0][0] - x)**2) + (point[0][1] - y)**2)
        #     dist = np.append(dist, distance)
        # maxima = argrelmax(dist, order = 25)
        return None

    elif(fingerLocation == "b"):
        """Todo work on finding fingers on the bottom"""

    elif(fingerLocation == "l"):
        for point in cntData:
            angle = math.degrees(math.atan2(point[0][1] - y, point[0][0] - x))
            deg = np.append(deg, angle)
            if (point[0][0] > centerX + 30):
                dist = np.append(dist, 0)
                continue
            distance = math.sqrt(((point[0][0] - x)**2) + (point[0][1] - y)**2)
            dist = np.append(dist, distance)
        maxima = argrelmax(dist, order = 25)
        prominence = peak_prominences(dist, maxima[0])
        fingers = []
        for finger in range(len(maxima[0])):
            if(prominence[0][finger] < 35):
                type = "closed"

            else:
                type = "open"

            fingers.append([finger, type])

        if (cv2.waitKey(1) == ord("w")):
            print(fingers)
        return fingers

    return None

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
    fgMask = cv2.GaussianBlur(fgMask, (3, 3), 0)
    fgMask = cv2.erode(fgMask, kernel, iterations = 2)
    hand = cv2.dilate(fgMask, kernel, iterations = 2)
    _, contours, _ = cv2.findContours(hand.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for x in range(1): #I'm sorry this is bad but I just need to break out at a specific point still in the loop
        if(contours):
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            area = cv2.contourArea(cnt)
            if((area > 20000) & (area < 55000)):
                M = cv2.moments(cnt)
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                cv2.circle(box, (x, y), 3, (0, 0, 255), -1)
                cv2.drawContours(box, [cnt], 0, (0, 0, 255), 2)
                if ((x < 150) | (x > 200) | (y < 225) | (y > 275)):
                    cv2.putText(frame, "Try to match the red dot to the blue dot", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness = 2)
                    continue

                cntTop = cnt.copy()
                deg = np.zeros(0)
                dist = np.zeros(0)
                for point in cntTop:
                    angle = math.degrees(math.atan2(point[0][1] - y, point[0][0] - x))
                    deg = np.append(deg, angle)
                    # if (point[0][1] > y + 30):
                    #     dist = np.append(dist, 0)
                    #     continue
                    # distance = math.sqrt(((point[0][0] - x)**2) + (point[0][1] - y)**2)
                    dist = np.append(dist, math.sqrt(((point[0][0] - x)**2) + (point[0][1] - y)**2))

                # fingers = []
                #open
                maxO, _ = find_peaks(dist, height=30, distance=5, prominence=30)
                if(len(maxO) > 4):
                    continue

                print(len(maxO))
                #closed
                maxC, _ = find_peaks(dist, prominence=(5, 25))
                if (len(maxC) > 4):
                    continue

                # print(len(maxC))

                # maxima = argrelmax(dist, order = 25)
                # prominence = peak_prominences(dist, maxima[0])

                #define states here
            elif(area >= 55000):
                cv2.putText(frame, "Hand is too close", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness = 2)

            elif(area > 10000):
                cv2.putText(frame, "Hand is too far", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness = 2)

    cv2.circle(box, (175, 250), 3, (255, 0, 0), -1)
    cv2.rectangle(frame, (760, 50), (1110, 450), (0, 255, 0), 1)
    cv2.imshow('original', frame)
    if(cv2.waitKey(1) == ord('q')):
        cap.release()
        break
    ret, frame = cap.read()
