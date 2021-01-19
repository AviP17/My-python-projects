import numpy as np
import cv2

cap = cv2.VideoCapture("Cafe Bene Intersection 2.mp4")


corners = 500

# params for ShiTomasi corner detection
feature_params = dict( maxCorners=corners, # The more corners that there are, the more points of motion are shown
                       qualityLevel = 0.1, # The quality level is the threshold (0-1) for motion detection
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( # A smaller window size means that the points can "stick" to objects better,
                  # but they're more sensitive to noise
                  winSize = (31, 31),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create a random color array that is as long as the number of corners
color = np.random.randint(0,255,(corners,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
initialp0 = np.copy(p0)    # Records the original point array as the default array
initialcorners = len(p0)   # This is the actual amount of points ont he screen

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
mask_array = []

while(1):
    ret,frame = cap.read()   # Reads image from the frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts image to grayscale

    # If too many points exit the screen, reset the initial points on the screen
    if len(p0) < (initialcorners*.97):
        p0 = initialp0
        # Alternative code
        '''corners = initialcorners - len(p0)
        extendedp0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        p0 = np.concatenate((p0, extendedp0), axis=0)
        np.unique(p0, axis=0)'''


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]


    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        # mask = cv2.line(mask, (a,b),(c,d), [0,255,0], 2)
        # frame = cv2.circle(frame,(a,b),5,[0,255,0],-1)

    # creates a list of active points that refreshes every frame
    # The first element of the list is subtracted from the mask so that the line length does not exceed 10
    if len(mask_array) < 10:
        mask_array.append(mask)
    else:
        mask_array.pop(0)
        mask = cv2.subtract(mask, mask_array[0])
        mask_array.append(mask)

    #display the combination of the mask and the frame
    img = cv2.add(frame, mask)
    cv2.imshow('frame',img)

    #break if the user presses the q key
    k = cv2.waitKey(30) & 0xff
    if k == ord("q"):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)


cv2.destroyAllWindows()
cap.release()