import os
import cv2
import numpy as np
import darknet
from collections import deque
from imutils.video import FPS
import argparse

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBall(detection, img):
    x, y, w, h = detection[2][0],\
        detection[2][1],\
        detection[2][2],\
        detection[2][3]
    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)

    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)

    cv2.putText(img,detection[0].decode() +" [" + str(round(detection[1] * 100, 2)) + "]",(pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
    return img

def getIDHighestDetection(detections):
    idOfMaxProbability = 0
    maxProbability = 0

    for index, detection in enumerate(detections):

        probability = detection[1]
        if(probability>maxProbability):
            maxProbability = probability
            idOfMaxProbability = index

    return idOfMaxProbability

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

netMain = None
metaMain = None
altNames = None

bufferSize = 200
pathToFile = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--mqtthost", default='localhost', help="hostname of the mqtt broker")
ap.add_argument("-p", "--mqttport", type=int, default=1883, help="port of the mqtt broker")
ap.add_argument("-v", "--video", default='empty', help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=200, help="max buffer size for lightning track")
ap.add_argument("-i", "--camindex", default=0, type=int, help="index of camera")
ap.add_argument("-c", "--color", default='0,0,0,0,0,0', help="not neccessary here, but important for java processbuilder")
ap.add_argument("-r", "--record", default='empty', help="switch on recording with following file name")
ap.add_argument("--showvideo", help="if true the video window is shown)")
args = vars(ap.parse_args())


if args["video"] is not 'empty':
    pathToFile = args["video"]
else:
    pathToFile = args["camindex"]

if args["buffer"] is not 'empty':
    bufferSize = args["buffer"]

if args["record"] is not 'empty':
    fileName = args["record"]

pts = deque(maxlen=bufferSize)


def YOLO():

    global metaMain, netMain, altNames
    configPath = 'obj.cfg'
    weightPath = 'obj.weights'
    metaPath = 'obj.data'
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass


    # read video file
    cap = cv2.VideoCapture(pathToFile)
    input_width = int(cap.get(3))
    input_height = int(cap.get(4))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Found Video: {input_width}x{input_height} - {input_fps} FPS" )


    # create output stream
    if fileName is not 'empty':
        out = cv2.VideoWriter((str(fileName)+'.mp4'), cv2.VideoWriter_fourcc(*'X264'), input_fps, (input_width, input_height))


    network_width = darknet.network_width(netMain)
    network_height = darknet.network_height(netMain)
    print(f"Network: {network_width}x{network_height}")
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(network_width, network_height, 3)

    scale_x = input_width / network_width
    scale_y = input_height / network_height

    print("Starting the YOLO loop...")
    frame_count = 0
    fps = FPS().start()
    while True:

        frame_count += 1

        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(frame_rgb,
                                   (network_width, network_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        position = (-1,-1)


        if not (len(detections) is 0):
            idOfDetection = getIDHighestDetection(detections)
            position = (int(detections[idOfDetection][2][0]), int(detections[idOfDetection][2][1]))
            pts.appendleft(position)
        #else:
        #    pts.append(None)

        # update fps every 10 frames
        if frame_count % 10 == 0:
            percent = 100 / input_frames * frame_count
            print(f"Ball: {position[0]},{position[1]} - {percent}% done - {fps.fps()} FPS")


        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(200 / float(i + 1)) * 2)
            cv2.line(frame_read, pts[i - 1] * scale_x, pts[i] * scale_y, (0, 255, 0), thickness)



        if args["record"] is not 'empty':
            out.write(frame_read)

        if args["showvideo"]: 
           cv2.imshow('Demo', frame_read)
           cv2.moveWindow("Demo", 1025,490)

        fps.update()
        cv2.waitKey(3)

    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
