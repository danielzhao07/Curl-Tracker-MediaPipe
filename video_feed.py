import cv2 as cv

# -- Video Feed --
cap = cv.VideoCapture(0) # setting up video capture device / int represents the recording device like webcam
while cap.isOpened(): # while loop - while cap is open / live feed
    ret, frame = cap.read() # ret is return variable, frame is the image - cap.read() reads the feed 
    cv.imshow('Mediapipe Feed', frame) # pop up window to visualize the image (frame)

    if cv.waitKey(10) & 0xFF == ord('q'): # break loop if window is closed or q is pressed
        break

cap.release() # release the video feed
cv.destroyAllWindows() # close all the windows