# Vision Manipulation & YOLOv8n-Seg Pipeline

This section covers the process of collecting an image dataset using an RGB-D camera, annotating and generating a dataset for instance segmentation via Roboflow, training and testing a YOLOv8n-seg model, and estimating the 3D position and rotation (grasping point) of objects for vision-based pick-and-place manipulation.[cite: 1]

---

## 1. Object Grasping Point & Rotation Estimation 

Vision-based robotic manipulation requires extracting precise spatial position and orientation from 2D/3D image data.

* **Vision Model Role**: Uses YOLOv8 nano segmentation (YOLOv8n-seg) to perform simultaneous object detection and instance segmentation on target objects.
* **Pose Extraction**: Leverages the predicted instance segmentation masks and pixel coordinates to infer the target object's pixel location and orientation (rotation angle) for robot grasping.

---

## 2. Data Collection for YOLOv8n Training ([collect_img.py](Data%20Collection%20for%20YOLOv8n%20Training/collect_img.py)) 

Acquires paired RGB color images and matching depth maps using the RealSense D435 camera to construct a dataset for model training.

* **Dataset Scale & Variations**: Collects a total of 200 images across 4 different objects (50 images per object) under various spatial positions and rotational angles.
* **Synchronized Saving**: Captures aligned color frames (`cv2.imwrite`) and raw 16-bit depth frames (`.astype(np.uint16)`) simultaneously.
* **User Control & Interface**: Displays real-time horizontally stacked RGB and depth colormap streams (`np.hstack`). Pressing `Spacebar` saves the current frame pair into `/color` and `/depth` subdirectories with indexed filenames (e.g., `cube_000.png`), while `ESC` terminates the program.

---

## 3. Dataset Generation for YOLOv8n via Roboflow 

Processes and labels raw images using the Roboflow web platform to prepare an instance segmentation dataset in YOLOv8 format.

* **Project Setup & Upload**: Creates a new project named "cube" configured for "Instance Segmentation" and uploads the collected images via drag-and-drop.
* **Polygon Annotation**: Uses the "Smart Select" tool to automatically create polygon borders around target objects (adjusting vertices manually if needed) and assigns the class label `cube`.
* **Exporting Data**: Generates dataset versions with Preprocessing and Augmentation options disabled. Selects the "YOLOv8" format, copies the generated download code/script, and executes it in the terminal to download the formatted dataset.

---

## 4. YOLOv8n-Seg Model Training ([train_yolo.py](train_yolo.py)) 
Trains the instance segmentation neural network using the prepared dataset and PyTorch framework.

* **Environment Preparation**: Requires installing PyTorch prior to running the training script.
* **Execution**: Runs the training script (`train_yolo.py`) to train the YOLOv8n-seg model on the custom-labeled `cube` dataset.

---

## 5. YOLOv8n-Seg Model Testing ([test_yolo.py](test_yolo.py))

Validates the performance of the trained instance segmentation model on sample images or live streams.

* **Inference & Verification**: Runs `test_yolo.py` to evaluate the trained model's ability to detect objects, generate segmentation masks, and output bounding boxes/rotation details.

---

## 6. Vision-Based Object Pick-and-Place ([pick_and_place_cube.py](pick_and_place_cube.py))

Integrates the vision model outputs with the robot control class to execute automated pick-and-place operations.

* **Coordinate Integration**: Translates the object's pixel mask coordinates and inferred rotation angle into 3D task space coordinates relative to the robot base frame using camera intrinsics and the extrinsic transformation matrix.
* **Execution**: Commands the robot arm (`movel`/`movej`) to approach the calculated 3D coordinates, align its gripper orientation to the inferred rotation angle, actuate the digital IO gripper (`grasp`/`release`), and transport the target object to its destination.
