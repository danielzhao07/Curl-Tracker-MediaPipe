import mediapipe as mp
import numpy as np
import cv2 as cv

mp_drawing = mp.solutions.drawing_utils # drawing utilites / visualizing poses
mp_pose = mp.solutions.pose # grabbing the pose estimation model

cap = cv.VideoCapture(0) # setting up video capture device / int represents the recording device like webcam
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: # set up new instance of mediapipe feed, access Pose estimation model, pass in two key arguements: detection confidence and tracking confidence. leverage as the variable pose
    while cap.isOpened(): # while loop - while cap is open / live feed
        ret, frame = cap.read() # ret is return variable, frame is the image - cap.read() reads the feed 
        
        # Recolor image to RGB for results
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # re-order the colour arrays
        image.flags.writeable = False # save memory by setting to false

        # Make detection
        results = pose.process(image)

        # Recolour back to RGB
        image.flags.writeable = True # set writeable back to true
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # re-render back to BGR, since opencv reads BGR

        print(results)

        # Render detections
        

        cv.imshow('Mediapipe Feed', frame) # pop up window to visualize the image (frame)

        if cv.waitKey(10) & 0xFF == ord('q'): # break loop if window is closed or q is pressed
            break

    cap.release() # release the video feed
    cv.destroyAllWindows() # close all the windows