
import datetime
import imutils
import cv2

# initialize the first frame in the video stream
vs = cv2.VideoCapture("Cafe Bene Intersection 2.mp4")


firstFrame = None
avg = None

# There are tradeoffs in computer vision
# If you want the data to capture less noise, you can increase the scope of the guassian kernel at the expense of
# missing important small details
GaussianWindow = (21, 21)

# If you want to capture subtler differences in the video, you can decrease the thresh value at the expense of
# increased noise
threshold = 1

# If you want have more stable boxes, you can increase minimum contour area at the expense of missing smaller objects
minContour = 800

# If you want to capture slow objects, you can decrease alpha at the expense of getting huge boxes for fast objects
alpha = .8


# loop over the frames of the video
while vs.isOpened():
    # grab the current frame and initialize the occupied/unoccupied
    # text
    frame = vs.read()
    frame = frame[1]
    text = "Unoccupied"
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
    # resize the frame, convert it to grayscale, and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, GaussianWindow, 0)
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    # compute the absolute difference between the current frame and
    # first frame
    if avg is None:
        avg = gray.copy().astype("float")

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, alpha)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    thresh = cv2.threshold(frameDelta, threshold, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < minContour:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
        # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # shows the frames
    # cv2.imshow("Security Feed", frame)
    cv2.imshow("Threshold", thresh)
    # cv2.imshow("Frame Delta", frameDelta)
    # cv2.imshow("Background", cv2.convertScaleAbs(avg))
    key = cv2.waitKey(20) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()