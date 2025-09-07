# Shape and Color Detection with OpenCV

This project uses **OpenCV** and **NumPy** to detect colored shapes in images and videos, find their centroids, and also compute approximate 3D coordinates from camera parameters.

## Repository Structure

- **Part1.py** – Detects colored shapes in a **static image** (`PennAir 2024 App Static.png`).  
- **Part2.py** – Detects colored shapes in a **video** (`PennAir 2024 App Dynamic.mp4`).  
- **Part3.py** – Detects shapes in a **video with background masking** (`PennAir 2024 App Dynamic Hard.mp4`) and marks centroids.  
- **Part4.py** – Extends Part 3 to estimate **3D coordinates** of detected circles using camera calibration values.  

## Requirements

- Python 3.8+
- OpenCV (https://opencv.org/) 
- NumPy


**Usage**

1. Static Shape Detection (Part1)
python Part1.py


Loads PennAir 2024 App Static.png.

Detects shapes of colors: red, yellow, green, blue, pink.

Displays contours and centroids.

2. Dynamic Shape Detection (Part2)
python Part2.py


Loads PennAir 2024 App Dynamic.mp4.

Detects colored shapes frame by frame.

Displays bounding contours and centroids in real-time.

3. Background Masking with Centroids (Part3)
python Part3.py


Loads PennAir 2024 App Dynamic Hard.mp4.

Uses HSV background sampling and masking to isolate shapes.

Draws contours and marks shape centroids.

4. 3D Coordinate Estimation (Part4)
python Part4.py


Loads PennAir 2024 App Dynamic Hard.mp4.


Computes approximate 3D position (X, Y, Z) of detected shapes.

Displays coordinates on output.

Controls

Press q while the video window is active to quit.








Input files (.png and .mp4) must be placed in the same directory as the scripts or updated with the correct path.

HSV color ranges and background thresholds may require adjustments for different lighting conditions.

