import numpy as np
import cv2

cap = cv2.VideoCapture('../../../Schreibtisch/testvideos/test_logitech1.avi')


onlyOnePerTwentyFrames = 0
i = 0

def click_and_crop(event, x, y, flags, param):
    global x_start, y_start, cropping
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start = x, y
        cropping = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        x_start, y_start = x, y
        
    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
    
cv2.namedWindow("cam1")
cv2.setMouseCallback("cam1", click_and_crop)

x_start=0;
y_start=0;
cropping=False;

while(cap.isOpened()):

    ret, frame = cap.read()

    x_treshold =  np.random.randint(100,150)
    y_treshold =  np.random.randint(100,150)
    
    x_end = x_start+x_treshold
    y_end = y_start+y_treshold
    
    if(x_end > 639):
        x_end = 639
    if(y_end > 479):
        y_end = 479
    
    upper_left = (x_start, y_start)
    bottom_right = (x_end, y_end)
    
    tmp_image=frame[upper_left[1]+2 : bottom_right[1]-2, upper_left[0]+2 : bottom_right[0]-2]
    cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), 2)
    
    cv2.imshow("cam1", frame)
    
    if(cropping == True):    
        out = cv2.imwrite("../../../Schreibtisch/testvideos/frames-high/vid-4-%s.jpg" % i, tmp_image)
        i= i+1

    
    if cv2.waitKey(200) & 0xFF == ord('q'):
        print("done")
        break

cap.release()
cv2.destroyAllWindows()
