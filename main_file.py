import mediapipe as mp
import numpy as np
import cv2 as cv

mp_drawing = mp.solutions.drawing_utils # drawing utilites / visualizing poses
mp_pose = mp.solutions.pose # grabbing the pose estimation model

# -- Video Feed --
cap = cv.VideoCapture(0) # setting up video capture device / int represents the recording device like webcam
while cap.isOpened(): # while loop - while cap is open / live feed
    ret, frame = cap.read() # ret is return variable, frame is the image - cap.read() reads the feed 
    cv.imshow('Mediapipe Feed', frame) # pop up window to visualize the image (frame)

    if cv.waitKey(10) & 0xFF == ord('q'): # break loop if window is closed or q is pressed
        break

cap.release() # release the video feed
cv.destroyAllWindows() # close all the windows