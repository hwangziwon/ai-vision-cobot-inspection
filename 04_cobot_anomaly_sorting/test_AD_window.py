import os
import cv2
import torch
import numpy as np

from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
from torchvision import transforms
from ultralytics import YOLO

from camera import RealSenseD435

# 학습할 때 사용한 RD4AD 모델 구조 파일
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2


# ============================================================
# 1. 설정
# ============================================================

YOLO_MODEL_PATH = "/home/sht/runs/segment/train-2/weights/best.pt"
ANOMALY_MODEL_PATH = "/home/sht/wres50_cube.pth"

YOLO_CONFIDENCE = 0.6

# 현재 테스트 점수 기준 임시값
# 정상/불량 데이터를 여러 장 테스트한 뒤 다시 조정해야 함
ANOMALY_THRESHOLD = 0.98

# ResNet에 실제 입력되는 crop 이미지 저장 여부
SAVE_DEBUG_CROP = True
DEBUG_CROP_DIR = "/home/sht/debug_crop"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if SAVE_DEBUG_CROP:
    os.makedirs(DEBUG_CROP_DIR, exist_ok=True)


# ============================================================
# 2. 이상탐지 입력 전처리
# ============================================================
#
# 학습할 때 YOLO crop 이미지를 바로 224×224로 변환했다는 전제
#
# 학습 코드에서 Resize(256) + CenterCrop(224)를 썼다면
# 이 부분도 똑같이 변경해야 함
# ============================================================

anomaly_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 3. RD4AD Wide-ResNet50 모델 불러오기
# ============================================================

def load_anomaly_model(model_path, device):
    print("\n=== Loading anomaly detection model ===")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Anomaly model file not found: {model_path}"
        )

    # pretrained encoder와 bottleneck 생성
    encoder, bottleneck = wide_resnet50_2(
        pretrained=True
    )

    # decoder 생성
    decoder = de_wide_resnet50_2(
        pretrained=False
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The .pth checkpoint is not a dictionary."
        )

    if "bn" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain the 'bn' key."
        )

    if "decoder" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain the 'decoder' key."
        )

    # 일부 RD4AD 버전에서 존재할 수 있는 memory 관련 키 제거
    bn_state_dict = checkpoint["bn"]

    for key in list(bn_state_dict.keys()):
        if "memory" in key:
            del bn_state_dict[key]

    bottleneck.load_state_dict(
        bn_state_dict,
        strict=True
    )

    decoder.load_state_dict(
        checkpoint["decoder"],
        strict=True
    )

    encoder = encoder.to(device)
    bottleneck = bottleneck.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    bottleneck.eval()
    decoder.eval()

    print("Anomaly model loaded successfully.")
    print(f"Device: {device}")

    return encoder, bottleneck, decoder


# ============================================================
# 4. Encoder와 Decoder feature 차이 계산
# ============================================================

def calculate_anomaly_map(
    encoder_features,
    decoder_features,
    output_size=224
):
    """
    encoder feature와 decoder feature의 cosine distance를 이용해
    anomaly map을 계산한다.

    반환값:
        anomaly_map: 224×224 NumPy 배열
    """

    anomaly_map = np.zeros(
        (output_size, output_size),
        dtype=np.float32
    )

    if len(encoder_features) != len(decoder_features):
        raise ValueError(
            "The number of encoder and decoder features is different."
        )

    for encoder_feature, decoder_feature in zip(
        encoder_features,
        decoder_features
    ):
        # cosine similarity가 낮을수록 feature 차이가 큼
        feature_anomaly_map = 1.0 - F.cosine_similarity(
            encoder_feature,
            decoder_feature,
            dim=1
        )

        feature_anomaly_map = feature_anomaly_map.unsqueeze(1)

        feature_anomaly_map = F.interpolate(
            feature_anomaly_map,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=True
        )

        feature_anomaly_map = (
            feature_anomaly_map[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        anomaly_map += feature_anomaly_map

    return anomaly_map


# ============================================================
# 5. 큐브 한 개에 대해 이상탐지
# ============================================================

def predict_anomaly(
    cube_crop,
    encoder,
    bottleneck,
    decoder,
    device
):
    """
    cube_crop:
        YOLO 바운딩 박스에서 잘라낸 OpenCV BGR 이미지

    반환값:
        anomaly_score
        anomaly_map
        resized_debug_image
    """

    if cube_crop is None or cube_crop.size == 0:
        return None, None, None

    # OpenCV BGR → RGB
    cube_rgb = cv2.cvtColor(
        cube_crop,
        cv2.COLOR_BGR2RGB
    )

    cube_pil = Image.fromarray(cube_rgb)

    # 학습 때와 동일하게 224×224 및 정규화
    input_tensor = anomaly_transform(
        cube_pil
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        # encoder feature 추출
        encoder_features = encoder(input_tensor)

        # bottleneck
        bottleneck_features = bottleneck(
            encoder_features
        )

        # decoder feature 복원
        decoder_features = decoder(
            bottleneck_features
        )

        anomaly_map = calculate_anomaly_map(
            encoder_features=encoder_features,
            decoder_features=decoder_features,
            output_size=224
        )

    # anomaly map 노이즈 완화
    anomaly_map = gaussian_filter(
        anomaly_map,
        sigma=4
    )

    # 단일 최대 픽셀은 노이즈에 민감하므로
    # anomaly 값이 높은 상위 1% 픽셀의 평균을 사용
    flattened_map = anomaly_map.flatten()

    top_pixel_count = max(
        1,
        int(flattened_map.size * 0.01)
    )

    top_scores = np.partition(
        flattened_map,
        -top_pixel_count
    )[-top_pixel_count:]

    anomaly_score = float(
        np.mean(top_scores)
    )

    # 실제 모델 입력 형태를 확인하기 위한 이미지
    resized_debug_image = cv2.resize(
        cube_crop,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    return anomaly_score, anomaly_map, resized_debug_image


# ============================================================
# 6. 바운딩 박스에 여유 공간 추가
# ============================================================

def expand_bbox(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
    margin_ratio=0.05
):
    """
    YOLO 박스가 큐브 가장자리를 너무 타이트하게 자르는 경우를 대비해
    바운딩 박스에 약간의 여백을 추가한다.

    학습 데이터가 박스 여백 없이 만들어졌다면
    margin_ratio를 0.0으로 설정하는 것이 좋다.
    """

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = int(box_width * margin_ratio)
    margin_y = int(box_height * margin_ratio)

    expanded_x1 = max(0, x1 - margin_x)
    expanded_y1 = max(0, y1 - margin_y)
    expanded_x2 = min(image_width, x2 + margin_x)
    expanded_y2 = min(image_height, y2 + margin_y)

    return (
        expanded_x1,
        expanded_y1,
        expanded_x2,
        expanded_y2
    )


# ============================================================
# 7. Depth 값 보완
# ============================================================

def get_valid_depth(
    depth_image,
    center_x,
    center_y,
    search_radius=3
):
    """
    중심 픽셀의 depth가 0인 경우 주변 픽셀에서
    유효한 depth 값의 중앙값을 계산한다.
    """

    image_height, image_width = depth_image.shape[:2]

    x1 = max(0, center_x - search_radius)
    y1 = max(0, center_y - search_radius)
    x2 = min(image_width, center_x + search_radius + 1)
    y2 = min(image_height, center_y + search_radius + 1)

    depth_region = depth_image[y1:y2, x1:x2]

    valid_depth_values = depth_region[
        np.isfinite(depth_region) & (depth_region > 0)
    ]

    if valid_depth_values.size == 0:
        return 0.0

    return float(
        np.median(valid_depth_values)
    )


# ============================================================
# 8. Anomaly map 시각화 및 저장
# ============================================================

def save_anomaly_debug_images(
    cube_index,
    original_crop,
    resized_crop,
    anomaly_map
):
    if not SAVE_DEBUG_CROP:
        return

    original_path = os.path.join(
        DEBUG_CROP_DIR,
        f"cube_{cube_index}_original.jpg"
    )

    resized_path = os.path.join(
        DEBUG_CROP_DIR,
        f"cube_{cube_index}_224.jpg"
    )

    anomaly_path = os.path.join(
        DEBUG_CROP_DIR,
        f"cube_{cube_index}_anomaly.jpg"
    )

    cv2.imwrite(
        original_path,
        original_crop
    )

    cv2.imwrite(
        resized_path,
        resized_crop
    )

    if anomaly_map is not None:
        anomaly_min = float(np.min(anomaly_map))
        anomaly_max = float(np.max(anomaly_map))

        if anomaly_max > anomaly_min:
            normalized_map = (
                (anomaly_map - anomaly_min)
                / (anomaly_max - anomaly_min)
            )
        else:
            normalized_map = np.zeros_like(
                anomaly_map,
                dtype=np.float32
            )

        normalized_map = (
            normalized_map * 255
        ).astype(np.uint8)

        anomaly_heatmap = cv2.applyColorMap(
            normalized_map,
            cv2.COLORMAP_JET
        )

        cv2.imwrite(
            anomaly_path,
            anomaly_heatmap
        )


# ============================================================
# 9. YOLO 검출 + ResNet 이상탐지
# ============================================================

def detect_cubes_once(
    camera,
    yolo_model,
    encoder,
    bottleneck,
    decoder,
    device,
    anomaly_threshold
):
    print(
        "\n=== Start YOLO detection "
        "and ResNet anomaly inspection ==="
    )

    color_image, depth_image = camera.get_image()

    if color_image is None:
        raise RuntimeError(
            "Failed to receive the color image."
        )

    if depth_image is None:
        raise RuntimeError(
            "Failed to receive the depth image."
        )

    # --------------------------------------------------------
    # YOLO 실행
    # --------------------------------------------------------
    results = yolo_model(
        color_image,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    result = results[0]

    annotated_frame = color_image.copy()

    detected_cubes = []

    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        print("No cube detected.")

        cv2.imshow(
            "YOLO + ResNet Cube Inspection",
            annotated_frame
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return detected_cubes

    image_height, image_width = color_image.shape[:2]

    # --------------------------------------------------------
    # 검출된 각 큐브 처리
    # --------------------------------------------------------
    for index, box in enumerate(boxes):
        xyxy = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )

        original_x1 = int(xyxy[0])
        original_y1 = int(xyxy[1])
        original_x2 = int(xyxy[2])
        original_y2 = int(xyxy[3])

        # 좌표가 이미지 범위를 벗어나지 않도록 제한
        original_x1 = max(
            0,
            min(original_x1, image_width - 1)
        )

        original_y1 = max(
            0,
            min(original_y1, image_height - 1)
        )

        original_x2 = max(
            1,
            min(original_x2, image_width)
        )

        original_y2 = max(
            1,
            min(original_y2, image_height)
        )

        if (
            original_x2 <= original_x1
            or original_y2 <= original_y1
        ):
            print(
                f"Cube[{index}]: invalid bounding box."
            )
            continue

        # 학습 데이터 crop 방식에 따라 조정
        #
        # 학습 때 YOLO 박스를 그대로 잘랐다면:
        # margin_ratio=0.0
        #
        # 학습 때 약간의 주변 영역이 포함됐다면:
        # margin_ratio=0.05
        x1, y1, x2, y2 = expand_bbox(
            original_x1,
            original_y1,
            original_x2,
            original_y2,
            image_width,
            image_height,
            margin_ratio=0.0
        )

        center_x = int(
            (original_x1 + original_x2) / 2
        )

        center_y = int(
            (original_y1 + original_y2) / 2
        )

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------
        distance_m = get_valid_depth(
            depth_image=depth_image,
            center_x=center_x,
            center_y=center_y,
            search_radius=3
        )

        # ----------------------------------------------------
        # YOLO 박스 영역 crop
        # ----------------------------------------------------
        cube_crop = color_image[
            y1:y2,
            x1:x2
        ].copy()

        # ----------------------------------------------------
        # ResNet 이상탐지
        # ----------------------------------------------------
        (
            anomaly_score,
            anomaly_map,
            resized_crop
        ) = predict_anomaly(
            cube_crop=cube_crop,
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            device=device
        )

        # ----------------------------------------------------
        # GOOD / BAD 판정
        # ----------------------------------------------------
        if anomaly_score is None:
            quality_label = "UNKNOWN"
            box_color = (0, 255, 255)

        elif anomaly_score >= anomaly_threshold:
            quality_label = "BAD"
            box_color = (0, 0, 255)

        else:
            quality_label = "GOOD"
            box_color = (0, 255, 0)

        # 디버깅 이미지 저장
        if resized_crop is not None:
            save_anomaly_debug_images(
                cube_index=index,
                original_crop=cube_crop,
                resized_crop=resized_crop,
                anomaly_map=anomaly_map
            )

        # ----------------------------------------------------
        # 세그멘테이션 마스크 기반 회전각
        # ----------------------------------------------------
        angle = 0.0
        rotated_box_points = None

        if (
            masks is not None
            and index < len(masks.xy)
        ):
            segment = masks.xy[index]

            if (
                segment is not None
                and len(segment) >= 3
            ):
                contour = segment.astype(
                    np.float32
                )

                rectangle = cv2.minAreaRect(
                    contour
                )

                rect_width, rect_height = (
                    rectangle[1]
                )

                raw_angle = float(
                    rectangle[2]
                )

                # OpenCV 버전에 따라 minAreaRect 각도 범위가 다를 수 있음
                if rect_width < rect_height:
                    angle = raw_angle + 90.0
                else:
                    angle = raw_angle

                rotated_box_points = cv2.boxPoints(
                    rectangle
                ).astype(np.int32)

        # ----------------------------------------------------
        # 결과 저장
        # ----------------------------------------------------
        detected_cubes.append({
            "index": index,
            "pixel_x": center_x,
            "pixel_y": center_y,
            "distance_z": distance_m,
            "rotation": float(angle),
            "anomaly_score": anomaly_score,
            "quality": quality_label,
            "bbox": [
                original_x1,
                original_y1,
                original_x2,
                original_y2
            ]
        })

        # ----------------------------------------------------
        # 결과 화면 표시
        # ----------------------------------------------------
        cv2.rectangle(
            annotated_frame,
            (original_x1, original_y1),
            (original_x2, original_y2),
            box_color,
            3
        )

        if rotated_box_points is not None:
            cv2.drawContours(
                annotated_frame,
                [rotated_box_points],
                0,
                box_color,
                2
            )

        cv2.circle(
            annotated_frame,
            (center_x, center_y),
            5,
            box_color,
            -1
        )

        if anomaly_score is None:
            result_text = quality_label
        else:
            result_text = (
                f"{quality_label} "
                f"{anomaly_score:.4f}"
            )

        label_y = max(
            original_y1 - 12,
            25
        )

        cv2.putText(
            annotated_frame,
            result_text,
            (original_x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2,
            cv2.LINE_AA
        )

        information_text = (
            f"Z:{distance_m:.3f}m "
            f"R:{angle:.1f}deg"
        )

        info_y = min(
            original_y2 + 22,
            image_height - 10
        )

        cv2.putText(
            annotated_frame,
            information_text,
            (original_x1, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # 결과 창
    # --------------------------------------------------------
    cv2.imshow(
        "YOLO + ResNet Cube Inspection",
        annotated_frame
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # 터미널 출력
    # --------------------------------------------------------
    print(
        f"\n[Detected] Number of cubes: "
        f"{len(detected_cubes)}"
    )

    print("------------------------------------------")

    for cube in detected_cubes:
        anomaly_score = cube["anomaly_score"]

        if anomaly_score is None:
            score_text = "None"
        else:
            score_text = f"{anomaly_score:.4f}"

        print(
            f"Cube[{cube['index']}] -> "
            f"Pixel X: {cube['pixel_x']}, "
            f"Pixel Y: {cube['pixel_y']}, "
            f"Distance Z: {cube['distance_z']:.3f}m, "
            f"Rotation: {cube['rotation']:.1f}°, "
            f"Anomaly Score: {score_text}, "
            f"Threshold: {anomaly_threshold:.4f}, "
            f"Result: {cube['quality']}"
        )

    if SAVE_DEBUG_CROP:
        print(
            f"\nDebug crop images saved to: "
            f"{DEBUG_CROP_DIR}"
        )

    return detected_cubes


# ============================================================
# 10. Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    # --------------------------------------------------------
    # YOLO 모델
    # --------------------------------------------------------
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}"
        )

    yolo_model = YOLO(
        YOLO_MODEL_PATH
    )

    # --------------------------------------------------------
    # Wide-ResNet50 RD4AD 모델
    # --------------------------------------------------------
    encoder, bottleneck, decoder = load_anomaly_model(
        model_path=ANOMALY_MODEL_PATH,
        device=DEVICE
    )

    # --------------------------------------------------------
    # RealSense 카메라
    # --------------------------------------------------------
    camera = RealSenseD435(
        color_resolution=720,
        depth_mode="720P"
    )

    try:
        cube_list = detect_cubes_once(
            camera=camera,
            yolo_model=yolo_model,
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            device=DEVICE,
            anomaly_threshold=ANOMALY_THRESHOLD
        )

        return cube_list

    finally:
        if hasattr(camera, "_pipeline"):
            camera._pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()