import cv2  # Loads the CV2 library for image manipulation.
import numpy

# Load video into project
cap = cv2.VideoCapture("PennAir 2024 App Dynamic Hard.mp4") # Creates a video capture object that will return the frames one by one.

# Background HSV sample (value found using a separate program to sample color from the video)
bg_color = numpy.array([0, 0, 51])

# Camera  matrix values
fx = 2564.3186869  # X focal length
fy = 2569.70273111 # Y focal length
cx, cy = 0, 0      # Principal point offsets
R = 10             # Actual circle radius

while True:
    ret, frame = cap.read() # Loop will only run if there are more frames left in the video.
    if not ret:
        break

    output = frame.copy() # Creates another frame copy to make changes to rather than editing the original frame.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converting to HSV with CV2 "Convert Color" function.

    # Creates a mask for the background. Then invert it to keep only the shapes.
    lower = numpy.array([bg_color[0], max(bg_color[0], 0), max(bg_color[2] - 51, 0)]) # Lower color limit for background mask.
    upper = numpy.array([bg_color[0] + 179, min(bg_color[0] + 55, 255), min(bg_color[2] + 100, 255)]) # Upper color limit for background mask.

    bg_mask = cv2.inRange(hsv, lower, upper)  # Mask for background.
    mask = cv2.bitwise_not(bg_mask)  # Inverts mask to keeps shapes only.

    # Removes speckles and smooths edges of shapes
    kernel = numpy.ones((7, 7), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) # Closes small gaps inside shapes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2) # Removes small noise around shapes.

    # Finds boundaries around the white shapes in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # RETR_EXTERNAL only creates contours around the footprint of the shapes.
    # CHAIN_APPROX_SIMPLE reduces the number of points in the contour to make it less noisy.

    # Loops through every contour and will calculate the areas.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 9000 or area > 80000:  # Ensures its a legit shape not ie background speckles or giant blobs
            continue

        # Will classify/simplify closed contour geometries based on number of sides on the closed shapes.
        approx = cv2.approxPolyDP(cnt, 0.00001 * cv2.arcLength(cnt, True), True)

        # Identify circle by number of vertices
        if len(approx) > 8:  # Circle has many sides when approximated
            cv2.drawContours(output, [approx], -1, (0, 200, 0), 3)

        # Draw contour function. Produces the actual visualization.
        cv2.drawContours(output, [approx], -1, (0, 200, 0), 3)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx_px = int(M["m10"] / M["m00"])  # Center in X direction
            cy_px = int(M["m01"] / M["m00"])  # Center in Y direction

            # Center dot and label.
            cv2.circle(output, (cx_px, cy_px), 7, (0, 0, 255), -1)
            cv2.putText(output, "center", (cx_px - 30, cy_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Estimate the 3D position using the equations.
            (circle_x, circle_y), r_px = cv2.minEnclosingCircle(cnt)  # Find smallest circle around the shape.
            r_px = max(r_px, 1)  # Prevents division by zero.

            Z = (fx * R) / r_px  # Depth
            X = (cx_px - cx) * Z / fx  # X into 3D.
            Y = (cy_px - cy) * Z / fy  # Y into 3D.

            # Display the 3D coordinates below the shape.
            x, y, w, h = cv2.boundingRect(cnt)
            label3D = f"3D Coords: ({X:.1f}, {Y:.1f}, {Z:.1f}) in"
            cv2.putText(output, label3D, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # Show result
    cv2.imshow("Mask", mask) # Displays the mask output only.
    cv2.imshow("Detected Shapes", output) # Displays the video output with shapes detected.

    # Quit when q is pressed.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # Closes everything.
