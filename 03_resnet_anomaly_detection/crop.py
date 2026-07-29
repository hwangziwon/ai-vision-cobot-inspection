import os
import cv2
from ultralytics import YOLO


# ============================================================
# 1. Path settings
# ============================================================

MODEL_PATH = "/home/sht/runs/segment/train-2/weights/best.pt" # use yolo model to find cube

INPUT_ROOT = "/home/sht/cube_raw_dataset"

# RD_Trainer.ipynb에서 바로 사용할 수 있도록 저장할 경우:
# OUTPUT_ROOT = "/path/to/03_resnet_anomaly_detection/data/cube"
OUTPUT_ROOT = "/home/sht/cube_crop_dataset"


# ============================================================
# 2. Crop settings
# ============================================================

CONF_THRESHOLD = 0.4

# Final square crop size relative to the longest bbox side
# 1.0: no additional margin
# 1.2: 20% larger than the longest bbox side
CROP_SCALE = 1.2

OUTPUT_SIZE = 224

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# 3. Dataset folders
# ============================================================

folders = [
    ("train/good", "train/good"),
    ("test/good", "test/good"),
    ("test/bad", "test/bad"),
]


# ============================================================
# 4. Load YOLO segmentation model
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# 5. Convert bbox to square crop
# ============================================================

def make_square_bbox(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
    scale=1.2,
):
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    box_width = x2 - x1
    box_height = y2 - y1

    square_size = max(box_width, box_height) * scale

    crop_x1 = int(center_x - square_size / 2.0)
    crop_y1 = int(center_y - square_size / 2.0)
    crop_x2 = int(center_x + square_size / 2.0)
    crop_y2 = int(center_y + square_size / 2.0)

    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(image_width, crop_x2)
    crop_y2 = min(image_height, crop_y2)

    return crop_x1, crop_y1, crop_x2, crop_y2


# ============================================================
# 6. Crop one image
# ============================================================

def crop_cube_image(image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread(image_path)

    if image is None:
        print(f"[READ FAILED] {image_path}")
        return False

    image_height, image_width = image.shape[:2]

    results = model(
        image,
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    if (
        len(results) == 0
        or results[0].boxes is None
        or len(results[0].boxes) == 0
    ):
        print(f"[NOT DETECTED] {image_path}")
        return False

    boxes = results[0].boxes

    # Select the detection with the highest confidence
    confidences = boxes.conf.detach().cpu().numpy()
    best_index = int(confidences.argmax())

    xyxy = (
        boxes.xyxy[best_index]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    x1, y1, x2, y2 = xyxy

    if x2 <= x1 or y2 <= y1:
        print(f"[INVALID BBOX] {image_path}")
        return False

    # Make the crop square and add margin
    crop_x1, crop_y1, crop_x2, crop_y2 = make_square_bbox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_width=image_width,
        image_height=image_height,
        scale=CROP_SCALE,
    )

    crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    if crop.size == 0:
        print(f"[CROP FAILED] {image_path}")
        return False

    crop_224 = cv2.resize(
        crop,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    filename = os.path.basename(image_path)
    save_path = os.path.join(save_dir, filename)

    success = cv2.imwrite(
        save_path,
        crop_224,
    )

    if not success:
        print(f"[SAVE FAILED] {save_path}")
        return False

    confidence = float(confidences[best_index])

    print(
        f"[SAVED] {save_path} "
        f"| confidence={confidence:.3f} "
        f"| crop={crop.shape[1]}x{crop.shape[0]}"
    )

    return True


# ============================================================
# 7. Process one dataset folder
# ============================================================

def process_folder(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"[FOLDER NOT FOUND] {input_dir}")
        return 0, 0

    filenames = sorted(os.listdir(input_dir))

    total_count = 0
    saved_count = 0

    for filename in filenames:
        extension = os.path.splitext(filename)[1].lower()

        if extension not in IMAGE_EXTENSIONS:
            continue

        total_count += 1

        image_path = os.path.join(
            input_dir,
            filename,
        )

        if crop_cube_image(
            image_path=image_path,
            save_dir=output_dir,
        ):
            saved_count += 1

    return total_count, saved_count


# ============================================================
# 8. Main
# ============================================================

def main():
    total_images = 0
    total_saved = 0

    for input_subdir, output_subdir in folders:
        input_dir = os.path.join(
            INPUT_ROOT,
            input_subdir,
        )

        output_dir = os.path.join(
            OUTPUT_ROOT,
            output_subdir,
        )

        print("\n========================================")
        print(f"Input : {input_dir}")
        print(f"Output: {output_dir}")
        print("========================================")

        image_count, saved_count = process_folder(
            input_dir=input_dir,
            output_dir=output_dir,
        )

        total_images += image_count
        total_saved += saved_count

        print(
            f"[FOLDER RESULT] "
            f"{saved_count}/{image_count} images saved"
        )

    print("\n========================================")
    print("Crop processing completed")
    print(f"Total images : {total_images}")
    print(f"Saved images : {total_saved}")
    print(f"Failed images: {total_images - total_saved}")
    print("========================================")


if __name__ == "__main__":
    main()
