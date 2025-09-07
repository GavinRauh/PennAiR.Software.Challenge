

import cv2
import numpy

# Load video into project
cap = cv2.VideoCapture("PennAir 2024 App Dynamic.mp4") # Creates a video capture object that will return the frames one by one.

# Define color ranges for the shapes (HSV - Hue, Saturation, Value).
color_ranges = {

# Wrote separate program to obtain shape color values in HSV.
    # Red - [  0 255 152]
    # Yellow - [ 33 190 255]
    # Green - [ 60 255 255]
    # Blue - [120 255 255]
    # Pink - [150 255 255]

    "red": [(0, 200, 100), (10, 300, 200)],
    "yellow": [(20, 100, 100), (35, 255, 255)],
    "green": [(45, 150, 150), (70, 255, 275)],
    "blue": [(100, 150, 0), (140, 255, 255)],
    "pink": [(140, 100, 100), (170, 255, 255)],

}


while True:
    ret, frame = cap.read() # Loop will only run if there are more frames left in the vid.
    if not ret:
        break

    output = frame.copy() # Creates another frame copy to make changes to rather than editing the original frame.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converting to HSV with CV2 "Convert Color" function.
    # BGR is not optimal bc it makes it difficult to classify what actually is "Red" or another color.

    for color, (lower, upper) in color_ranges.items(): # Loops through each color within the specified range

        # Create a mask
        mask = cv2.inRange(hsv, numpy.array(lower), numpy.array(upper))

        # Finds boundaries around the white shapes in the mask.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # RETR_EXTERNAL only creates contours around the footprint of shape.
        # CHAIN_APPROX_NONE does not simplify the contour geometry.



        # Loops through every contour and will calculate the areas.
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Ensures its a legit shape not ie background speckles.
                continue

            # Will classify/simplify closed contour geometries based on number of sides on the closed shapes.
            approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)

            # Draw contour function. Produces the actual visualization.
            cv2.drawContours(output, [approx], -1, (0, 200, 0), 3)

            # Compute centroid.
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx_px = int(M["m10"] / M["m00"])  # Center in X direction
                cy_px = int(M["m01"] / M["m00"])  # Center in Y direction
                cv2.circle(output, (cx, cy), 7, (0, 0, 255), -1)


    # Show result
    cv2.imshow("Detected Shapes", output)

    # Quit when q is pressed.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows() # Closes everything.
