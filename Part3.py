

import cv2
import numpy



# Load video into project
cap = cv2.VideoCapture("PennAir 2024 App Dynamic Hard.mp4")  # Creates a video capture object that will return the frames one by one.

# Wrote separate program to obtain background HSV sample
bg_color = numpy.array([0, 0, 51])

while True:
    ret, frame = cap.read()  # Loop will only run if there are more frames left in the vid.
    if not ret:
        break

    output = frame.copy()  # Creates another frame copy to make changes to rather than editing the original frame.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converting to HSV with CV2 "Convert Color" function.



    # Create a mask for the background. Then invert to isolate the shapes.
    lower = numpy.array([bg_color[0], max(bg_color[0], 0), max(bg_color[2]-51, 0)]) # Set the limits for the background mask
    upper = numpy.array([bg_color[0] + 179, min(bg_color[0] + 55, 255), min(bg_color[2] + 100, 255)]) # Set the limits for the background mask

    # Create the mask
    bg_mask = cv2.inRange(hsv, lower, upper)
    # Invert the mask so it masks anything that isn't the background
    mask = cv2.bitwise_not(bg_mask)

    # Morphological cleaning to reduce noise and fill gaps
    kernel = numpy.ones((7,7), numpy.uint8)  # bigger kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # stronger close
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # clean edges

    # Finds boundaries around the white shapes in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # RETR_EXTERNAL only creates contours around the footprint of shape.
    # CHAIN_APPROX_SIMPLE simplifies the contour geometry (less noisy than NONE).

    # Loops through every contour and will calculate the areas.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 9000 or area > 80000:  # Ensures its a legit shape not ie background speckles or giant blobs
            continue

        # Will classify/simplify closed contour geometries based on number of sides on the closed shapes.
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # Draw contour function. Produces the actual visualization.
        cv2.drawContours(output, [approx], -1, (0, 200, 0), 3)

        # Compute centroid.
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx_px = int(M["m10"] / M["m00"])  # Center in X direction
            cy_px = int(M["m01"] / M["m00"])  # Center in Y direction
            cv2.circle(output, (cx, cy), 7, (0, 0, 255), -1)

    # Show result
    cv2.imshow("Mask", mask) # Display the mask view.
    cv2.imshow("Detected Shapes", output) # Display the video output.

    # Quit when q is pressed.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  # Closes everything.
