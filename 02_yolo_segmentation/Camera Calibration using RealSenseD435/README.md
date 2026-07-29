# Camera Calibration using Intel RealSense

This guide covers the entire process of modularizing a camera control pipeline into an object-oriented (OOP) class using the Intel RealSense SDK 2.0 and OpenCV, from real-time image acquisition to calculating 3D robot target coordinates via inverse projection.

---

## 1. RealSense Camera Class Design & Modularization ([camera.py](camera.py))
Similar to the robot control class (`robot.py`), hardware control algorithms are encapsulated into a Python class structure to maximize reusability and scalability.

* **`__init__(self)` (Instance Initialization & Pipeline Setup)**: Creates kernel pipeline (`rs.pipeline()`) and configuration (`rs.config()`) objects from the `pyrealsense2` library.
* **`start(self)` (Start Streaming)**: Enables both RGB color and depth streams simultaneously, registering an `rs.align()` processor to align the depth pixels to the color lens coordinate frame and prevent pixel-matching errors.
* **`stop(self)` (Release Hardware Resources)**: Safely shuts down the active RealSense pipeline buffer (`pipeline.stop()`) on system termination to prevent hardware conflicts.

---

## 2. Real-Time RGB-D Image Acquisition & Dataset Collection ([capture.py](capture.py))
Synchronously accesses the device frame buffer (`wait_for_frames()`) to save high-quality image samples to disk for camera matrix calculation and calibration.

* **NumPy Matrix Conversion & OpenCV Output**: Parses incoming raw data buffers into 2D matrices (`numpy.asanyarray()`) readable by OpenCV (RGB and 16-bit depth matrix) and renders them in real time.
* **Keyboard Triggered Image Sampling**: Captures and saves the aligned color frame (`cv2.imwrite()`) and matching depth map data simultaneously as uniquely indexed files (e.g., `color_0.png`, `depth_0.png`) whenever the user presses the `Spacebar`.

---

## 3. Checkerboard-Based Intrinsic Calibration ([calibration.py](calibration/calibration.py))
Runs OpenCV geometric/math operations using the collected image dataset to compute lens distortion coefficients and the focal length matrix.

* **`cv2.findChessboardCorners()` (Grid Corner Detection)**: Loads saved checkerboard images to detect corner intersections and refines pixel coordinates to sub-pixel precision using `cv2.cornerSubPix()`.
* **`cv2.calibrateCamera()` (Intrinsic Matrix Derivation)**: Inputs the aligned spatial coordinates into the solver engine to compute the camera intrinsic matrix ($K$, containing focal lengths and principal points) and distortion coefficients ($D$), then serializes the results to JSON or YAML format.

---

## 4. Real-Time Mouse Pixel Coordinates & 3D Data Mapping ([read_pixel.py](transform%20coord/read_pixel.py))
Retrieves target pixel coordinates and their corresponding real depth values based on user mouse interaction in a live video stream.

* **OpenCV Mouse Callback Event (`cv2.setMouseCallback`)**: Registers mouse click events (`cv2.EVENT_LBUTTONDOWN`) on the color frame window to extract 2D image pixel coordinates $(u, v)$ in real time.
* **Real-Time Pixel Depth Matching**: Uses the clicked pixel coordinate $(u, v)$ as an index to query the aligned depth matrix, obtaining the physical distance $Z$ (in mm) from the camera lens to the target object and printing it to the terminal.

---

## 5. Robot Spatial Coordinate Calculation via Pixel Inverse Projection ([get_robot_coord.py](transform%20coord/get_robot_coord.py))
Combines the camera intrinsic matrix ($K$), pixel coordinates, and sensor depth values mathematically (inverse projection) to derive the 3D physical task coordinates for the Doosan robot.

* **3D Spatial Inverse Projection**: Passes the mouse click coordinates $(u, v)$ and depth $Z$ through the inverse projection formula using the focal lengths ($f_x, f_y$) and principal points ($c_x, c_y$). This derives the physical 3D coordinates $(X_c, Y_c, Z_c)$ in the camera coordinate frame, centered at the camera lens $(0,0,0)$.
  $$\quad X_c = \frac{(u - c_x) \times Z}{f_x} \quad,\quad Y_c = \frac{(v - c_y) \times Z}{f_y} \quad,\quad Z_c = Z$$
* **Camera-to-Robot Coordinate Transformation Matrix**: Applies a hand-eye calibration matrix (transformation matrix) to convert 3D coordinates from the camera frame to task space coordinates relative to the Doosan cobot base frame. This allows the robot to move (`movel`) accurately to the computed target position.
