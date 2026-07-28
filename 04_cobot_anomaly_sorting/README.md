## Model Preparation

**Project Demonstration Video:**  
https://mysterious-broccoli-188.notion.site/AI-3a7af7312d1d80c0aa91dc4cb98e3fdb?source=copy_link

The anomaly detection model used in the final robot system was prepared through the following workflow:

1. Detect cubes in the workspace using a **YOLO Segmentation** model.
2. Collect images of normal cubes using an **Intel RealSense** camera.
3. Crop the object region based on the YOLO segmentation mask or bounding box.
4. Train a **ResNet-based anomaly detection model** using the cropped images as the input dataset.
5. Integrate the trained model into the final collaborative robot classification system.

The related source code can be found in the following directories:

- **YOLO Detection:** `../02_yolo_segmentation`
- **Data Preprocessing and ResNet Training:** `../03_resnet_anomaly_detection`
