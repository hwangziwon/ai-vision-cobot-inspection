# AI Vision-Based Cobot Inspection System

This repository summarizes an AI vision and collaborative robot project completed as part of the **WISET AI Robot System Engineer Training Program**.

Starting with CNN-based image classification, the project progressed through YOLO object detection and ResNet-based anomaly detection. Finally, the vision AI models were integrated with a Doosan collaborative robot to build an automated inspection system that classifies products as either good or defective.

**Demonstration Video:**  
https://mysterious-broccoli-188.notion.site/AI-3a7af7312d1d80c0aa91dc4cb98e3fdb?source=copy_link

---

# Project Overview

This repository consists of the following four projects:

```text
01. CNN-Based Image Classification
02. YOLO-Based Object Detection and Localization
03. ResNet-Based Anomaly Detection
04. AI Vision-Based Automated Sorting with a Collaborative Robot
```

The models and functionalities developed in each project were integrated into the final collaborative robot inspection system.

---

# System Workflow

```text
RealSense Camera Capture
        ↓
YOLO-Based Cube Detection
        ↓
Cube Center, Depth, and Rotation Estimation
        ↓
ResNet-Based Anomaly Detection
        ↓
Camera-to-Robot Coordinate Transformation
        ↓
Doosan Cobot Pick-and-Place
        ↓
Automatic Sorting of Good and Defective Products
```

---

# Repository Structure

```text
ai-vision-cobot-inspection/
│
├── 01_cnn_classification/
│   └── CNN-based image classification
│
├── 02_yolo_segmentation/
│   ├── ROS2-Based Doosan Cobot Control
│   ├── Camera Calibration using RealSenseD435
│   └── Robotic Vision Manipulation

│
├── 03_resnet_anomaly_detection/
│   ├── RealSense data collection
│   ├── YOLO-based cube cropping
│   ├── 224 × 224 dataset generation
│   └── Training and evaluation of a ResNet-based anomaly detection model
│
├── 04_cobot_anomaly_sorting/
│   ├── Automated sorting system integrating AI vision and a collaborative robot
│   ├── Integration of YOLO and ResNet with the Doosan cobot
│   ├── Cube position, orientation, and anomaly detection
│   └── Automatic GOOD/BAD sorting
│
├── assets/
│   └── Project images and demonstration materials
│
└── README.md
```

---

# 01. CNN Classification

This project was conducted to understand the fundamentals of CNNs and the image classification process.

- Image data preprocessing
- CNN model implementation
- Model training and evaluation
- Inference on new images

---

# 02. YOLO Segmentation

A YOLO Segmentation model was trained and applied to detect cubes captured by an Intel RealSense camera.

Main features include:

- Cube detection
- Instance mask extraction
- Center pixel calculation
- Depth estimation
- Cube rotation angle estimation

---

# 03. ResNet Anomaly Detection

A ResNet-based model trained with normal cube images was used to determine whether an input image was normal or anomalous.

```text
Input Image
     ↓
YOLO Object Cropping
     ↓
ResNet Feature Extraction
     ↓
Anomaly Score Calculation
     ↓
GOOD / NG Classification
```

The anomaly score was compared with a predefined threshold to classify products as either good or defective.

---

# 04. Cobot Anomaly Sorting

This is the final AI robot project, integrating the YOLO and ResNet models with a **Doosan Robotics M0609** collaborative robot.

## Main Features

- RealSense D435 image acquisition
- YOLO-based cube detection
- ResNet-based anomaly detection
- Camera-to-robot coordinate transformation
- Robot grasping based on cube position and orientation
- Automatic sorting of good and defective products
- Compliance control during the center gear assembly process

---

# Tech Stack

## AI / Vision

- Python
- PyTorch
- Ultralytics YOLO
- CNN
- ResNet
- OpenCV
- NumPy

## Robot / Sensor

- Doosan Robotics M0609
- ROS 2
- Intel RealSense D435
- Pick-and-Place
- Force Control
- Compliance Control

---

# Project Result

The completed system recognizes objects using a vision sensor, determines whether they are normal or defective using AI models, and automatically sorts them into different locations using a collaborative robot.

The entire workflow is integrated into a single system:

```text
Perception
      ↓
AI-Based Inspection
      ↓
Coordinate Transformation
      ↓
Robot Manipulation
      ↓
Automatic Sorting
```

---

# Notes

The trained model weights and the complete dataset are not included in this repository due to file size limitations and dataset management considerations.
