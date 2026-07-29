"""
{AI 비전 기반 협동로봇 큐브 자동 분류 시스템}

주요 처리 과정
---------------
1. RealSense D435로 작업 영역 촬영
2. YOLO Segmentation으로 큐브 검출
3. ResNet 기반 이상 탐지
4. 카메라 좌표를 로봇 좌표로 변환
5. GOOD/BAD 판정에 따른 자동 분류

{주의사항}
--------
- 모델 경로와 Place 위치는 실행 환경에 맞게 수정해야 합니다.
- 실제 로봇 실행 전 작업 영역과 좌표값을 검증해야 합니다.
"""
import rclpy
import DR_init
import sys
import numpy as np
import time
# move_home, move_j, move_l, grasp, release 등의 래퍼 함수를 제공
from robot import Robot
# Ultralytics YOLO 모델을 불러오기 위한 클래스
from ultralytics import YOLO
from camera import RealSenseD435

# YOLO 검출과 ResNet 기반 이상 탐지 기능이 구현된 사용자 정의 모듈
from test_AD_window import (
    # 카메라 영상 한 프레임을 기준으로 큐브를 검출하고,
    # 각 큐브의 위치, 깊이, 회전각, 이상 점수, 품질 판정 결과를 반환
    detect_cubes_once,
    load_anomaly_model,
    DEVICE,
    ANOMALY_MODEL_PATH,
)

# ============================================================
# 1. YOLO 모델 설정
# ============================================================

# 학습이 완료된 YOLO Segmentation 모델의 가중치 파일 경로
# 이 경로는 현재 Ubuntu 실행 환경을 기준으로 하므로,
# 다른 컴퓨터에서 실행할 경우 실제 best.pt 파일 위치로 변경해야 함
YOLO_MODEL_PATH = "/home/sht/runs/segment/train-2/weights/best.pt"

# YOLO 모델을 한 번만 메모리에 로드합니다.
# 이후 detect_cubes_once() 함수에 전달해 반복적으로 사용할 수 있습니다.
yolo_model = YOLO(YOLO_MODEL_PATH)

# ============================================================
# 2. Camera 좌표계 → Robot 좌표계 외부 파라미터
# ============================================================

# 카메라 좌표계에서 측정한 3차원 점을
# 로봇 기준 좌표계(Base 또는 World 좌표계)로 변환하기 위한
# 4×4 동차 변환 행렬(Homogeneous Transformation Matrix)
#
# 행렬의 구조는 다음과 같습니다.
#
# [ R11 R12 R13 Tx ]
# [ R21 R22 R23 Ty ]
# [ R31 R32 R33 Tz ]
# [  0   0   0   1 ]
#
# - 왼쪽 위 3×3 행렬 R: 카메라와 로봇 좌표계 사이의 회전 관계
# - 오른쪽 3×1 벡터 T: 카메라 원점과 로봇 원점 사이의 평행이동 관계
#
# 현재 행렬의 단위는 미터(m)를 기준으로 함.
# 따라서 변환 결과를 두산로봇 명령에 사용할 때는 mm로 변환함.
extrinsic_matrix = np.array([
    [
        -9.842804480632064701e-01,
        9.256802314739166759e-02,
        -1.504099752379392252e-01,
        -4.134985284882139456e-01,
    ],
    [
        1.695416795147855760e-01,
        7.338012390910285676e-01,
        -6.578688018258492809e-01,
        3.353550706617078325e-01,
    ],
    [
        4.947341172588870517e-02,
        -6.730281588455686581e-01,
        -7.379603505156259180e-01,
        2.408146706989155172e-01,
    ],
    [
        0.0,
        0.0,
        0.0,
        1.0,
    ],
], dtype=np.float32)

# ============================================================
# 3. 로봇 기본 자세와 순응제어 강성 설정
# ============================================================

# 각 큐브의 작업을 시작하거나 종료할 때 이동하는 로봇 관절 자세
# 두산로봇의 Joint 좌표 형식: [J1, J2, J3, J4, J5, J6]
# 단위는 degree
task_home_joint = [
    -180.00,
    0.00,
    90.00,
    0.00,
    90.00,
    60.00,
]

# 순응제어(Compliance Control)에서 사용할 강성값
# 일반적인 순서:[X축, Y축, Z축, Rx축, Ry축, Rz축]
# 값이 클수록 해당 방향으로 단단하게 버티고,
# 값이 작을수록 외력에 더 쉽게 순응

# 여기서는 Z축 강성을 상대적으로 낮게 설정하여
# 로봇이 큐브를 향해 하강할 때 충돌 힘을 감지하면 다시 팔을 올리도록 함
stiffness = [
    3000,
    3000,
    150,
    200,
    200,
    200,
]

# ============================================================
# 4. Pick 작업 높이 설정
# ============================================================

# 큐브 바로 위에 위치하는 안전 대기 높이
# 로봇의 Task 좌표계 기준 Z 값, 단위는 mm
PICK_READY_Z = -180.0

# 큐브를 실제로 파지하기 위해 내려갈 최종 목표 높이
# 로봇은 이 높이까지 한 번에 이동하지 않고 5mm씩 나누어 하강
# 직접교시를 통해 값을 알아두면 좋음
PICK_Z = -231.0

# ============================================================
# 5. GOOD / BAD Place 위치
# ============================================================

# 이상 탐지 결과가 GOOD인 큐브를 놓을 위치
# 실제 작업대에서 티칭한 로봇 좌표로 반드시 변경해야 함
GOOD_PLACE_X = -425.0
GOOD_PLACE_Y = -300.0
GOOD_PLACE_Z = -180.0

# 이상 탐지 결과가 BAD인 큐브를 놓을 위치
BAD_PLACE_X = -425.0
BAD_PLACE_Y = -150.0
BAD_PLACE_Z = -180.0

def main(args=None):
    """
    전체 AI 비전 및 로봇 제어 파이프라인을 실행합니다.

    실행 순서
    --------
    1. ROS 2 초기화
    2. 두산로봇 ID와 모델 설정
    3. 로봇 제어 노드 생성
    4. 이상 탐지 모델 로드
    5. RealSense 카메라 초기화
    6. YOLO 검출 및 이상 탐지 수행
    7. 검출된 큐브를 순차적으로 Pick & Place
    8. GOOD/BAD 판정에 따라 서로 다른 위치에 배치
    9. 카메라, 순응제어, ROS 2 자원 정리
    """

    # ROS 2 통신 시스템을 초기화
    rclpy.init(args=args)

    # 사용 중인 두산로봇의 네임스페이스 ID
    ROBOT_ID = "dsr02"=
    ROBOT_MODEL = "m0609"

    # DSR_ROBOT2 모듈이 어떤 로봇과 통신할지 알 수 있도록
    # 로봇 ID와 모델명을 DR_init 전역 변수에 저장
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    # 두산로봇 명령을 송수신할 ROS 2 노드를 생성
    # namespace를 ROBOT_ID로 설정해 해당 로봇 네임스페이스와 연결
    node = rclpy.create_node(
        "example_py",
        namespace=ROBOT_ID
    )

    # 생성한 ROS 2 노드를 두산로봇 초기화 모듈에 등록
    DR_init.__dsr__node = node

    # DR_init 설정이 완료된 후 DSR_ROBOT2 함수를 import합니다.
    # 두산로봇 예제에서는 로봇 ID, 모델, 노드 설정 이후 import하는 구조를 사용합니다.
    from DSR_ROBOT2 import (
        amovel,
        set_robot_mode,
        ROBOT_MODE_AUTONOMOUS,
        DR_TOOL,
        task_compliance_ctrl,
        release_compliance_ctrl,
        get_current_pose,
        check_force_condition,
        DR_AXIS_Z,
    )
    set_robot_mode(
        ROBOT_MODE_AUTONOMOUS
    )
    robot = Robot(node)

    camera = None
    try:
        # ----------------------------------------------------
        # 6. 로봇 초기 위치 이동
        # ----------------------------------------------------

        # 두산로봇에 미리 설정된 Home 자세로 이동
        # 인자 1은 Robot 클래스 내부에서 정의한 Home 종류 또는 모드를 의미
        robot.move_home(1)

        # ----------------------------------------------------
        # 7. ResNet 기반 이상 탐지 모델 로딩
        # ----------------------------------------------------

        # 이상 탐지 네트워크를 구성하는 세 부분을 불러옴
        #
        # encoder:
        #   입력 이미지에서 특징을 추출
        #
        # bottleneck:
        #   추출된 특징을 압축하거나 중간 표현으로 변환
        #
        # decoder:
        #   특징을 복원하여 정상 데이터와의 차이를 계산하는 데 사용
        encoder, bottleneck, decoder = load_anomaly_model(
            model_path=ANOMALY_MODEL_PATH,
            device=DEVICE
        )

        # ----------------------------------------------------
        # 8. RealSense 카메라 초기화
        # ----------------------------------------------------
        # color_resolution=720:
        #   컬러 이미지 해상도를 720P로 사용
        #
        # depth_mode="720P":
        #   깊이 영상도 720P 모드로 사용
        camera = RealSenseD435(
            color_resolution=720,
            depth_mode="720P"
        )
        # ----------------------------------------------------
        # 9. YOLO 검출 + ResNet 이상 탐지
        # ----------------------------------------------------

        # 카메라에서 한 번 촬영한 영상으로 큐브들을 검출
        #
        # 반환되는 cube_list의 각 원소는 일반적으로 다음 정보를 포함함
        #
        # {
        #     "pixel_x": 큐브 중심의 X 픽셀 좌표,
        #     "pixel_y": 큐브 중심의 Y 픽셀 좌표,
        #     "distance_z": 카메라로부터 큐브까지의 깊이[m],
        #     "rotation": 큐브 회전각[degree],
        #     "quality": "GOOD" 또는 "BAD",
        #     "anomaly_score": 이상 점수
        # }
        cube_list = detect_cubes_once(
            camera=camera,
            yolo_model=yolo_model,
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            device=DEVICE,

            # 이상 점수가 이 임계값을 기준으로 GOOD/BAD로 판정
            anomaly_threshold=0.98 # 모델의 성능에 따라 조절 가능
        )

        # 검출된 큐브 수를 출력
        print(
            f"\nRobot will process "
            f"{len(cube_list)} cubes."
        )

        # ----------------------------------------------------
        # 10. 검출된 큐브를 하나씩 순차 처리
        # ----------------------------------------------------

        # enumerate()를 사용해 큐브 인덱스 i와 큐브 정보 cube를 함께 가져옴
        for i, cube in enumerate(cube_list):
            print(
                "\n======================================"
            )
            print(
                f"Start processing Cube[{i}]"
            )

            # 각 큐브 작업 시작 전에 기준 관절 자세로 이동
            robot.move_j(
                task_home_joint
            )

            # 파지 전 그리퍼를 열어 둡니다.
            robot.release()

            # ----------------------------------------------
            # 10-1. 이상 탐지 결과와 카메라 측정값 읽기
            # ----------------------------------------------

            # 큐브 중심의 이미지 픽셀 X 좌표
            center_x = cube["pixel_x"]

            # 큐브 중심의 이미지 픽셀 Y 좌표
            center_y = cube["pixel_y"]

            # 카메라에서 큐브까지의 깊이값
            cam_z = cube["distance_z"]

            # 영상에서 추정한 큐브의 회전각
            # 그리퍼의 회전 방향을 큐브 방향에 맞추는 데 사용
            yaw = cube["rotation"]

            # 이상 탐지 결과
            # 정상은 "GOOD", 이상은 "BAD"로 처리
            quality = cube["quality"]

            # 이상 정도를 나타내는 수치
            # anomaly_threshold와 비교해 품질 판정에 사용
            anomaly_score = cube["anomaly_score"]

            # 현재 큐브의 검출 및 판정 정보를 터미널에 출력합니다.
            print(
                f"Cube[{i}] -> "
                f"quality={quality}, "
                f"score={anomaly_score:.4f}, "
                f"pixel=({center_x}, {center_y}), "
                f"depth={cam_z:.3f}m, "
                f"yaw={yaw:.1f}deg"
            )

            # ----------------------------------------------
            # 10-2. 깊이값 유효성 검사
            # ----------------------------------------------

            # 깊이값이 NaN, Inf이거나 0 이하이면
            # 올바른 3차원 좌표를 계산할 수 없으므로 로봇 이동을 금지함
            #
            # 잘못된 깊이값으로 로봇을 이동시키면
            # 예상하지 못한 위치로 움직일 수 있으므로 중요한 안전 검사
            if (
                not np.isfinite(cam_z)
                or cam_z <= 0
            ):
                print(
                    f"Cube[{i}] skipped: "
                    f"invalid depth value {cam_z}"
                )
                continue

            # ----------------------------------------------
            # 10-3. 품질 판정값 유효성 검사
            # ----------------------------------------------

            # 품질 결과가 GOOD 또는 BAD가 아닌 경우에는
            # 분류 위치를 결정할 수 없으므로 해당 큐브를 건너뜀
            if quality not in (
                "GOOD",
                "BAD"
            ):
                print(
                    f"Cube[{i}] skipped: "
                    f"unknown quality '{quality}'"
                )
                continue

            # ----------------------------------------------
            # 10-4. 큐브 회전각 범위 보정
            # ----------------------------------------------

            # 큐브는 180도 회전해도 파지 방향이 동일하게 보일 수 있으므로
            # 회전각을 -90도에서 +90도 사이에 들어오도록 정규화
            #
            # 예:
            #  120도 → -60도
            # -120도 →  60도
            if yaw > 90:
                yaw -= 180

            elif yaw < -90:
                yaw += 180

            # ----------------------------------------------
            # 10-5. Pixel 좌표 + Depth → Camera 3D 좌표
            # ----------------------------------------------

            # 카메라 내부 파라미터 행렬에서 초점거리와 주점을 가져옴, 제공됨
            #
            # 내부 파라미터 행렬 형식:
            #
            # [ fx  0  cx ]
            # [  0 fy  cy ]
            # [  0  0   1 ]
            #
            # fx, fy:
            #   X, Y 방향 초점거리[pixel]
            #
            # cx, cy:
            #   카메라 영상의 주점(principal point)[pixel]
            fx = camera._color_intrinsics_mat[0][0]
            fy = camera._color_intrinsics_mat[1][1]
            cx = camera._color_intrinsics_mat[0][2]
            cy = camera._color_intrinsics_mat[1][2]

            # 핀홀 카메라 모델을 이용해 픽셀 X 좌표를
            # 카메라 기준 3차원 X 좌표로 역투영
            #
            # X = (u - cx) × Z / fx
            cam_x = (
                center_x - cx
            ) * cam_z / fx

            # 픽셀 Y 좌표를 카메라 기준 3차원 Y 좌표로 역투영
            #
            # Y = (v - cy) × Z / fy
            cam_y = (
                center_y - cy
            ) * cam_z / fy

            # 카메라 좌표계의 3차원 점 [X, Y, Z]를 생성
            cam_coordinate = np.array([
                cam_x,
                cam_y,
                cam_z,
            ], dtype=np.float32)

            # ----------------------------------------------
            # 10-6. Camera 좌표 → Robot World 좌표 변환
            # ----------------------------------------------

            # 외부 파라미터의 회전행렬 R을 카메라 좌표에 곱함
            #
            # P_robot = R × P_camera + T
            world_coordinate = (
                extrinsic_matrix[0:3, 0:3]
                @ cam_coordinate
            )

            # 회전된 좌표에 평행이동 벡터 T를 더해
            # 최종 로봇 기준 좌표를 계산
            world_coordinate += (
                extrinsic_matrix[0:3, 3]
            )

            # 계산된 로봇 좌표를 확인용으로 출력
            print(
                f"Cube[{i}] world coordinate:\n"
                f"{world_coordinate}"
            )

            # 외부 파라미터 결과는 meter 단위이므로
            # 두산로봇 명령에서 사용하는 millimeter 단위로 변환함
            robot_x = float(
                world_coordinate[0] * 1000
            )

            robot_y = float(
                world_coordinate[1] * 1000
            )

            # Z 좌표는 카메라 변환값을 직접 사용하지 않고
            # 사전에 티칭한 안전 대기 높이로 고정함
            robot_z = PICK_READY_Z

            # ----------------------------------------------
            # 10-7. 큐브 회전각에 맞게 그리퍼 Yaw 정렬
            # ----------------------------------------------

            # 현재 로봇의 Joint 자세를 가져와 리스트로 변환
            # 현재 코드에서는 get_current_pose(0)의 반환값을 Joint 자세로 사용
            prepare_gripper_yaw = list(
                get_current_pose(0)
            )

            # 6번째 관절(J6)의 각도를 큐브 회전각만큼 보정
            #
            # 현재 J6 각도에서 yaw를 빼는 이유는
            # 영상 좌표계의 회전 방향과 로봇 관절 회전 방향을 맞추기 위함
            prepare_gripper_yaw[5] = (
                prepare_gripper_yaw[5]
                - yaw
            )

            # 보정된 관절 자세로 이동하여
            # 그리퍼 방향을 큐브 방향과 정렬
            robot.move_j(
                prepare_gripper_yaw
            )

            # ----------------------------------------------
            # 10-8. Pick 준비 위치로 이동
            # ----------------------------------------------

            # 현재 로봇의 Task 자세를 가져옵니다.
            # 일반적인 Task pose 형식:
            # [X, Y, Z, Rx, Ry, Rz]
            grasp_ready_pose = list(
                get_current_pose(1)
            )

            # 계산한 큐브 X, Y 위치를 Task pose에 반영
            grasp_ready_pose[0] = robot_x
            grasp_ready_pose[1] = robot_y

            # Z는 실제 파지 위치보다 높은 안전 대기 높이로 설정
            grasp_ready_pose[2] = robot_z

            # 직선 이동으로 큐브 위 준비 위치까지 이동
            robot.move_l(
                grasp_ready_pose
            )

            # ----------------------------------------------
            # 10-9. Pick 하강 목표 자세 생성
            # ----------------------------------------------

            # 준비 자세를 복사하여 실제 파지 목표 자세를 생성
            grasp_pose = (
                grasp_ready_pose.copy()
            )

            # 최종 파지 높이로 Z 값만 변경
            grasp_pose[2] = PICK_Z

            # ----------------------------------------------
            # 10-10. 순응제어 활성화
            # ----------------------------------------------

            # 지정한 강성값으로 순응제어를 시작
            task_compliance_ctrl(
                stx=stiffness
            )

            # False로 시작하고 힘 조건이 감지되면 True로 변경
            collision_detected = False

            # 현재 준비 위치의 Z 값을 정수로 변환
            start_z = int(
                grasp_ready_pose[2]
            )

            # 최종 파지 목표 Z 값도 정수로 변환
            target_z = int(
                grasp_pose[2]
            )

            # ----------------------------------------------
            # 10-11. 5mm 단위 하강 및 충돌 검사
            # ----------------------------------------------

            # 한 번에 PICK_Z까지 이동하지 않고 5mm씩 나누어 하강합니다.
            #
            # 예:
            # PICK_READY_Z = -180
            # PICK_Z       = -231
            #
            # 실제 명령 Z:
            # -185, -190, -195, ... , -230

            for current_z in range(
                start_z - 5,
                target_z - 1,
                -5
            ):
                step_pose = (
                    grasp_ready_pose.copy()
                )
                step_pose[2] = float(
                    current_z
                )
                # 비동기 선형 이동 명령으로 5mm 하강
                # 비동기 명령을 사용하는 이유:
                # 로봇이 이동하는 동안 아래의 힘 조건 검사를 동시에 수행하기 위함
                amovel(
                    step_pose,
                    vel=10,
                    acc=10
                )

                t_start = time.time()

                while (
                    time.time() - t_start
                    < 0.2
                ):
                    # 주의:
                    #   check_force_condition()의 반환 규칙은
                    #   사용 중인 두산로봇 API 버전과 예제 코드를 반드시 확인해야 합니다.
                    force_result = (
                        check_force_condition(
                            axis=DR_AXIS_Z,
                            min=8,
                            ref=DR_TOOL
                        )
                    )

                    # 현재 기존 코드에서는 반환값 0을
                    # 힘 조건이 감지된 상태, 즉 충돌 발생으로 해석
                    if force_result == 0:
                        print(
                            f"Cube[{i}] "
                            "collision detected."
                        )

                        collision_detected = True
                        break

                # 충돌이 감지되었다면 더 이상 하강하지 않고
                # 5mm 단계 반복문을 종료
                if collision_detected:
                    break

            # ----------------------------------------------
            # 10-12. 충돌 감지 시 안전 복귀
            # ----------------------------------------------

            if collision_detected:
                # 순응제어를 종료하고 일반 위치제어 상태로 복귀
                release_compliance_ctrl()
                robot.move_l(
                    grasp_ready_pose
                )
                # 그리퍼를 열어 혹시 접촉한 물체를 놓음
                robot.release()

                # 기준 관절 자세로 복귀
                robot.move_j(
                    task_home_joint
                )

                # 현재 큐브 작업을 종료하고 다음 큐브로 넘어감
                continue

            # ----------------------------------------------
            # 10-13. 정상적으로 큐브 파지
            # ----------------------------------------------

            # 충돌 없이 하강이 완료되면 순응제어를 종료
            release_compliance_ctrl()

            # 그리퍼를 닫아 큐브를 파지
            robot.grasp()

            # 큐브를 파지한 상태로 준비 높이까지 다시 상승
            robot.move_l(
                grasp_ready_pose
            )

            # ----------------------------------------------
            # 10-14. 품질 판정에 따른 Place 위치 선택
            # ----------------------------------------------

            # 현재 Pick 준비 자세를 복사해 Place 자세의 기본값으로 사용
            # 자세 방향 Rx, Ry, Rz는 유지하고 X, Y, Z 위치만 변경
            place_pose = (
                grasp_ready_pose.copy()
            )

            # GOOD 판정인 경우 GOOD 전용 배치 위치를 적용
            if quality == "GOOD":
                place_pose[0] = GOOD_PLACE_X
                place_pose[1] = GOOD_PLACE_Y
                place_pose[2] = GOOD_PLACE_Z

                print(
                    f"Cube[{i}] GOOD -> "
                    "move to GOOD area"
                )

            else:
                place_pose[0] = BAD_PLACE_X
                place_pose[1] = BAD_PLACE_Y
                place_pose[2] = BAD_PLACE_Z

                print(
                    f"Cube[{i}] BAD -> "
                    "move to BAD area"
                )

            # ----------------------------------------------
            # 10-15. Place 위치로 이동 및 큐브 배치
            # ----------------------------------------------

            # 선택된 GOOD 또는 BAD 위치로 직선 이동
            robot.move_l(
                place_pose
            )
            robot.release()

            # ----------------------------------------------
            # 10-16. 배치 후 수직 방향으로 이탈
            # ----------------------------------------------

            # Place 자세를 복사해 안전 이탈 자세를 생성
            retreat_pose = (
                place_pose.copy()
            )
            # 현재 배치 위치에서 Z 방향으로 50mm 상승
            retreat_pose[2] += 50.0
            robot.move_l(
                retreat_pose
            )

        # ----------------------------------------------------
        # 11. 모든 큐브 작업 완료 후 기준 자세 복귀
        # ----------------------------------------------------
        # 모든 큐브 처리가 끝나면 기준 관절 자세로 복귀
        robot.move_j(
            task_home_joint
        )

        print(
            "\nAll cube sorting tasks completed."
        )

    finally:
        # ----------------------------------------------------
        # 12. 예외 발생 여부와 관계없이 실행할 종료 처리
        # ----------------------------------------------------

        # 카메라 객체가 정상 생성되었고 내부 pipeline이 존재하면
        # RealSense 스트리밍을 종료
        if camera is not None and hasattr(
            camera,
            "_pipeline"
        ):
            camera._pipeline.stop()

        try:
            release_compliance_ctrl()
        except Exception:
            pass
        
        robot.release()

        robot.move_j(
            task_home_joint
        )

        rclpy.shutdown()
        
if __name__ == "__main__":
    main()
