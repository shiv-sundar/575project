import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, peak_prominences, find_peaks

def checkHandClosed(cntData, centerX, centerY):

    # if (cntData[0][1])
    """Use distance from center to top of contour,
    if relatively small, then hand closed"""

def getAngles(cnt):
    deg = np.zeros(0)
    for point in cnt:
        angle = math.degrees(math.atan2(point[0][1] - y, point[0][0] - x))
        deg = np.append(deg, angle)

    return deg

def getDist(cnt):
    dist = np.zeros(0)
    for point in cnt:
        dist = np.append(dist, math.sqrt(((point[0][0] - x)**2) + (point[0][1] - y)**2))

    return dist

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
backSub = cv2.createBackgroundSubtractorKNN()
while(ret):
    frame = cv2.flip(frame, 1)
    box = frame[50:450, 710:1110, :]
    t = box.copy()
    t = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    fgMask = backSub.apply(t)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fgMask = cv2.GaussianBlur(fgMask, (3, 3), 0)
    fgMask = cv2.erode(fgMask, kernel, iterations = 2)
    hand = cv2.dilate(fgMask, kernel, iterations = 2)
    _, contours, _ = cv2.findContours(hand.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for x in range(1): #I'm sorry, this is bad, but I just need to break out at a specific point still in the loop
        if(contours):
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            area = cv2.contourArea(cnt)
            cv2.drawContours(box, [cnt], 0, (0, 0, 255), 2)
            if((area > 20000) & (area < 55000)):
                M = cv2.moments(cnt)
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                cv2.circle(box, (x, y), 3, (0, 0, 255), -1)
                if ((x < 175) | (x > 225) | (y < 225) | (y > 275)):
                    cv2.putText(frame, "Try to match the red dot to the blue dot", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness = 2)
                    continue

                first = np.argmax(cnt[:, :, 1]> 395)
                cnt2 = cnt[first:0:-1].copy()
                cnt2 = np.append(cnt2, cnt[cnt.shape[0] - 1:first:-1, :, :].copy(), axis=0)
                """Check hand closed first"""
                # this is letters: A, E, M, N, O, P, Q, S, T

                """Check open hand letters"""
                # letters: B, C, D, F, K, L, R, U, V, W, Y
                cntTop = cnt2.copy()
                distTop = np.zeros(0)
                for point in cntTop:
                    if (point[0][1] < 350):
                        distTop = np.append(distTop, math.sqrt(((point[0][0] - x)**2) + (point[0][1] - y)**2))

                # distTop = getDist(cntTop)

                #openTop
                maxO, dict = find_peaks(distTop, height=125, prominence=40, width=(None, None))
                if(len(maxO) > 5):
                    continue

                #closedTop
                maxC, _ = find_peaks(distTop, prominence=(2, 10))
                if (len(maxC) != 5 - len(maxO)):
                    continue

                # thumb
                cntThumb = cnt.copy()
                distT = np.zeros(0)

                #use custom angle measures
                for point in cntThumb:
                    angle = math.degrees(math.atan2(point[0][1] - y, point[0][0] - x))
                    if ((angle > 135) | (angle < -135)):
                        distT = np.append(distT, math.sqrt(((point[0][0] - x)**2) + (point[0][1] - y)**2))

                maxT, _ = find_peaks(distT, height=100, prominence=20, width=20)
                cv2.putText(frame, "Thumb is " + str(len(maxT)) + " and " + str(len(maxO)) + " fingers open", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness = 2)
                """Finding states here"""
                for x in range(1): """Make sure to take out this for loop"""
                    if ((len(maxO) == 1) & (len(maxT) == 0)):
                        if (dict["widths"] < 70):
                            if (maxO[0] < 185):
                                print("D")
                                continue

                            if (maxO[0] >= 185):
                                print("I")
                                continue

                        if (dict["widths"] >= 70):
                            print("B")
                            continue

                    if ((len(maxO) == 2) & (len(maxT) == 1)):
                        print("L")
                        continue

                    if ((len(maxO) == 2) & (len(maxT) == 0)):
                        if ((cnt2[maxO[0]][0][0] < box.shape[0]/2) &(cnt2[maxO[1]][0][0] < box.shape[0]/2) & (len(maxT) == 0)):
                            print("V")
                            continue

                    if ((len(maxO) == 3) & (len(maxT) == 0)):
                        print("F")
                        continue

                    if ((len(maxO) == 4)):
                        print("idiot")

                plt.plot(distTop)
                plt.show()

            elif(area >= 55000):
                cv2.putText(frame, "Hand is too close", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness = 2)

            elif(area > 10000):
                cv2.putText(frame, "Hand is too far", (0, 720), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness = 2)

    cv2.circle(box, (200, 250), 3, (255, 0, 0), -1)
    cv2.rectangle(frame, (710, 50), (1110, 450), (0, 255, 0), 1)
    cv2.imshow('original', frame)
    if(cv2.waitKey(1) == ord('q')):
        cap.release()
        break
    ret, frame = cap.read()
