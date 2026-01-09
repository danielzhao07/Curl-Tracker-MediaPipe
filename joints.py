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

        # Extract landmarks (connection points)
        try:
            landmarks = results.pose_landmarks.landmark # variable to hold landmarks, get landmarks with method
        except:
            pass # pass if no detections or error / step out of loop

        # use try and except block to avoiding destroy entire feed, pass if no detections are made

        # Render detections
        # draw detections to image
        # - use drawing utilities, pass the image, pass the result landmarks and pass the pose connections (which landmarks are connected to which)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
                                # ^^ added 2 drawing specs, specifications for our drawing components (landmarks)
                                # first spec is colour dots, second spec is the connections - coloured lines
        
        # display image with new landmarks drawn
        cv.imshow('Mediapipe Feed', image) # pop up window to visualize the image (frame)

        if cv.waitKey(10) & 0xFF == ord('q'): # break loop if window is closed or q is pressed
            break

    cap.release() # release the video feed
    cv.destroyAllWindows() # close all the windows

## - operators to find landmarks and landmarks values

len(landmarks) # prints 33 since there are 33 landmarks in the human body

for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk) # prints all kind of landmarks, eg: PoseLandmark.NOSE, PoseLandmark.____

print(mp_pose.PoseLandmark.NOSE.value) # prints index 0

landmarks[mp_pose.PoseLandmark.NOSE.value] # same as below
landmarks[0] # prints the coordinates (x, y, z) and visibility value