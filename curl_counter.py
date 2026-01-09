import mediapipe as mp
import numpy as np
import cv2 as cv
import angles

mp_drawing = mp.solutions.drawing_utils # drawing utilites / visualizing poses
mp_pose = mp.solutions.pose # grabbing the pose estimation model

cap = cv.VideoCapture(0) # setting up video capture device / int represents the recording device like webcam

# Curl counter variables
counter = 0
stage = None # whether at down or up position

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
            
            # Get coordinates (specifically x and y coords)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y] # store x and y coordinate of the left shoulder in a list
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] # store x and y coordinate of the elbow shoulder in a list
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] # store x and y coordinate of the wrist shoulder in a list
        
            # Calculate angle between the 3 landmarks
            angle = angles.calculate_angle(shoulder, elbow, wrist)

            # Visualize Angle on screen
            cv.putText(image, str(angle), 
                       tuple(np.multiply(elbow, [640, 480],).astype(int)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AAA)
            # ^ use cv2 putText method: pass in image, angle by converting to string, determine where positioning is (multiply elbow coordinate and dimensions of webcam) (convert to int then tuple, tuple holds x and y coordinate based on the webcam video)
            # font type, font size, colour of font (white), line width, line type

            # Curl counter logic
            if angle > 160: # if angle is greater than 160, then down position
                stage = 'down'
            if angle < 30 and stage == 'down': # if angle is less than 30 and stage is currently in down position
                stage = 'up' # reset to up
                counter += 1 # count 1 rep
                print(counter)

        except:
            pass # pass if no detections or error / step out of loop
        
        # use try and except block to avoiding destroy entire feed, pass if no detections are made

        # Render curl counter and visualize to screen
        # Setup status box
        cv.rectangle(image, (0,0), (225, 73), (245, 117, 16), thickness=-1) # image, start point of box, end point of box, colour, fill in square (-1 thickness)
        
        # Rep Data / Write in Box
        cv.putText(image, 'REPS', (15, 12), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA) # displays title

        cv.putText(image, str(counter), (10, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 1, cv.LINE_AA) # displays counter
        
        # Stage Data / Write in Box
        cv.putText(image, 'STAGE', (65, 12), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA) # displays title

        cv.putText(image, stage, (60, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 1, cv.LINE_AA) # displays stage

        # ^^ image, text to display, start coordinate, font, font size, colour, line width, line type

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

### ----- TESTING ------ ###

## - operators to find landmarks and landmarks values

len(landmarks) # prints 33 since there are 33 landmarks in the human body

for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk) # prints all kind of landmarks, eg: PoseLandmark.NOSE, PoseLandmark.____

print(mp_pose.PoseLandmark.NOSE.value) # prints index 0

landmarks[mp_pose.PoseLandmark.NOSE.value] # same as below
landmarks[0] # prints the coordinates (x, y, z) and visibility value

shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y] # store x and y coordinate of the left shoulder in a list
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] # store x and y coordinate of the elbow shoulder in a list
wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] # store x and y coordinate of the wrist shoulder in a list

print(angles.calculate_angle(shoulder, elbow, wrist)) # calculate the angle between shoulder, elbow and wrist by passing in the x and y coordinates of the 3 landmarks in the angles file

