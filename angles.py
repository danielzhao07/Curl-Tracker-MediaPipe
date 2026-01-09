import numpy as np

def calculate_angle(a, b, c): # fcn to calculate angle through 3 (first, mid, end) pts (landmark numbers)
    # convert to numpy arrays
    a = np.array(a) # first
    b = np.array(b) # second
    c = np.array(c) # third     

    # calculate radians to find angle, same with x value - do the same previous with first and mid points - (0, 1, 2) <- (x, y, z)
    # subtract y value from endpoint and midpoint
    radians = np.arctan2(c[1]-b[1]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    # calculate angle, convert radians to degrees
    angle = np.abs(radians*180.0/np.pi)

    # convert angle between 0 and 180 degrees
    if angle > 180.0:
        angle = 360 - angle
    
    # return our angle
    return angle