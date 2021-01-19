import cv2
import numpy as np
cap = cv2.VideoCapture("Cafe Bene Intersection 2.mp4") # Reads from webcam is parameter is zero

ret, frame1 = cap.read() # Initializes the first frame of the video
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Converts the image to grayscale
hsv = np.zeros_like(frame1) # A new frame is made where every pixel is initially black
hsv[...,1] = 255 # The saturation of every pixel is maximized

while(1):
    # Reads the next frame and converts it to grayscale
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Returns an cartesian vector array for each frame by calculating optical flow
    # Optical flow is calculated by calculating the movement of points of constant intensity'''
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # converts the vectors to polar form
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # The hue of each pixel is determined by the direction of each pixel's optical flow
    hsv[...,0] = ang*180/np.pi/2
    # The value of each pixel is determined by the magnitude of each pixel's optical flow
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)*(cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)>40)
    # The hue, saturation, and value of each pixel is converted to an RGB value
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    # Displays both the original frame and the processed frame

    cv2.imshow('frame2',rgb)
    #cv2.imshow('frame1', frame2)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()