

import cv2
import numpy

# Load image
img = cv2.imread("PennAir 2024 App Static.png")
output = img.copy()

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color ranges (in HSV)
color_ranges = {
    "red": [(0, 200, 100), (10, 300, 200)],
    "yellow": [(20, 100, 100), (35, 255, 255)],
    "green": [(45, 150, 150), (70, 255, 275)],
    "blue": [(100, 150, 0), (140, 255, 255)],
    "pink": [(140, 100, 100), (170, 255, 255)],
}

for color, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv, numpy.array(lower), numpy.array(upper))

    # Find contours for this color
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # skip tiny contours
            continue

        approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)

        # Draw contour function. Produces the actual visualization.
        cv2.drawContours(output, [approx], -1, (0, 200, 0), 2)

        # Compute center
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx_px = int(M["m10"] / M["m00"])  # Center in X direction
            cy_px = int(M["m01"] / M["m00"])  # Center in Y direction
            cv2.circle(output, (cx, cy), 5, (0, 0, 275), -1)

# Show result
cv2.imshow("Detected Shapes", output)
cv2.destroyAllWindows()
