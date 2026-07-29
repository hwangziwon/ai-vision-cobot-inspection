# 03. ResNet Anomaly Detection

This directory contains the data collection, preprocessing, training, and evaluation code for cube anomaly detection.

The model is trained only on normal cube images and identifies defective cubes based on differences from learned normal features.

Datasets and trained model weights are not included in this repository due to their file sizes.

---

## Workflow

```text
RealSense image collection
        ↓
Raw dataset organization
        ↓
YOLO-based cube detection and cropping
        ↓
Resize cropped images to 224 × 224
        ↓
Train with normal cube images
        ↓
Evaluate with normal and defective images
```

---

## Files

| File | Description |
|---|---|
| `realsense2.ipynb` | Collects full workspace images using a RealSense camera |
| `crop.py` | Detects cubes using the YOLO model from the previous stage and saves 224 × 224 crop images |
| `RD_Trainer.ipynb` | Trains the anomaly detection model using normal cube images |
| `RD_Tester.ipynb` | Evaluates normal and defective images and visualizes anomaly regions |
| `resnet.py` | Defines the Wide-ResNet50 encoder and bottleneck |
| `de_resnet.py` | Defines the decoder for feature reconstruction |

---

## Dataset Structure

### Raw Dataset

```text
cube_raw_dataset/
├── train/
│   └── good/
└── test/
    ├── good/
    └── bad/
```

### Cropped Dataset

```text
cube_crop_dataset/
├── train/
│   └── good/
└── test/
    ├── good/
    └── bad/
```

- `train/good`: normal images used for training
- `test/good`: normal images used for evaluation
- `test/bad`: defective images used for evaluation

---

## Usage

### 1. Collect Images

```bash
jupyter notebook realsense2.ipynb
```

Collect normal training images and normal/defective test images.

### 2. Crop Cube Images

```bash
python3 crop.py
```

The script uses the YOLO model trained in the previous stage to detect cubes, create square crops, and resize them to `224 × 224`.

Update the paths in `crop.py` before execution.

```python
MODEL_PATH = "/path/to/yolo/best.pt"
INPUT_ROOT = "/path/to/cube_raw_dataset"
OUTPUT_ROOT = "/path/to/cube_crop_dataset"
```

### 3. Train the Model

```bash
jupyter notebook RD_Trainer.ipynb
```

The model is trained using only images from:

```text
cube_crop_dataset/train/good/
```

### 4. Evaluate the Model

```bash
jupyter notebook RD_Tester.ipynb
```

The model is evaluated using:

```text
cube_crop_dataset/test/good/
cube_crop_dataset/test/bad/
```

---

## Model Output

The model generates an anomaly map and anomaly score for each image.

```text
Low anomaly score  → GOOD
High anomaly score → BAD
```

The anomaly score is not a probability. The classification threshold should be determined using normal and defective test samples.

---

## Notes

Training and inference preprocessing should use the same crop size, object scale, lighting conditions, and image normalization.

The following files are excluded from this repository:

```text
cube_raw_dataset/
cube_crop_dataset/
*.pth
```
