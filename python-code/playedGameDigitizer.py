from collections import deque
import cv2 as cv
import argparse
import imutils
import time

#well, comments instead of clean code. blame me (TODO)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

frameSize = 800

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=20000)

pts.appendleft((0, 0,time.time()))
pts.appendleft((0, 0,time.time()))

cap = cv.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resize the frame and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=frameSize)

    # Our operations on the frame come here
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV,  then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    # lower_blue = np.array([110,78,50])
    # upper_blue = np.array([130,255,255])
    mask = cv.inRange(hsv, greenLower, greenUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), time.time())


    # update the points queue
    pts.appendleft(center)

    if pts[0] is None or pts[1] is None:
        actualPointX, actualPointY = 0, 0
        previousPointX, previousPointY = 0, 0
    else:
        actualPointX = pts[1][0]
        actualPointY = pts[1][1]
        previousPointX = pts[0][0]
        previousPointY = pts[0][1]


    print(actualPointX,  actualPointY)
    print(pts.__sizeof__())

    # res = cv.bitwise_and(frame,frame, mask= mask)
    # Display the resulting frame
    cv.imshow('frame', frame)

    #key for quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        file = open("ballMovementInCoordinates.txt","w")

        pts.reverse()

        for i in range(0, len(pts)):
            file.write(repr(pts[i][2]) + "|" + repr(pts[i][0]) + "|" + repr(pts[i][1])+"\n")

        file.close()
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()