import rclpy
import DR_init
import sys
import numpy as np
import time

from robot import Robot
from ultralytics import YOLO
from camera import RealSenseD435
from test_yolo import detect_cubes_once

model_path = "/home/sht/runs/segment/train-2/weights/best.pt"
model = YOLO(model_path)

extrinsic_matrix = np.array([[-9.842804480632064701e-01, 9.256802314739166759e-02, -1.504099752379392252e-01, -4.134985284882139456e-01],
    [1.695416795147855760e-01, 7.338012390910285676e-01, -6.578688018258492809e-01, 3.353550706617078325e-01],
    [4.947341172588870517e-02, -6.730281588455686581e-01, -7.379603505156259180e-01, 2.408146706989155172e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]], dtype=np.float32)

task_home_joint = [-180.00, 0.00, 90.00, 0.00, 90.00, 60.00]

stiffness = [3000, 3000, 150, 200, 200, 200]

def main(args=None):
    rclpy.init(args=args)
    
    ROBOT_ID = "dsr02"
    ROBOT_MODEL = "m0609"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    
    node = rclpy.create_node('example_py', namespace=ROBOT_ID)
    
    DR_init.__dsr__node = node
    
    from DSR_ROBOT2 import amovel, set_robot_mode, ROBOT_MODE_AUTONOMOUS, DR_TOOL, task_compliance_ctrl, release_compliance_ctrl, get_current_pose, check_force_condition, DR_AXIS_Z
    
    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    
    robot = Robot(node)
    robot.move_home(1) # joint (0, 0, 90, 0, 90, 0)
    camera = RealSenseD435(color_resolution=720, depth_mode="720P")
    cube_list = detect_cubes_once(camera=camera, model=model)

    for i in range(len(cube_list)):
        robot.move_j(task_home_joint)
        robot.release()

        center_x = cube_list[i][0]
        center_y = cube_list[i][1]
        cam_z = cube_list[i][2]
        yaw = cube_list[i][3]

        if yaw > 90:
            yaw = yaw - 180

        cam_x = np.multiply(center_x - camera._color_intrinsics_mat[0][2], cam_z /
                            camera._color_intrinsics_mat[0][0])
        cam_y = np.multiply(center_y - camera._color_intrinsics_mat[1][2], cam_z /
                            camera._color_intrinsics_mat[1][1])

        cam_coordinate = [cam_x, cam_y, cam_z, yaw]

        world_coordinate = extrinsic_matrix[0:3, 0:3] @ cam_coordinate[0:3]
        world_coordinate += extrinsic_matrix[0:3, 3:].flatten()
        print(f"world_coord\n{world_coordinate}")

        robot_x = world_coordinate[0] * 1000 # meter to mm
        robot_y = world_coordinate[1] * 1000 # meter to mm
        robot_z = -180

        # Get joint value
        prepare_gripper_yaw = list(get_current_pose(0))
        prepare_gripper_yaw[5] = prepare_gripper_yaw[5] - yaw

        robot.move_j(prepare_gripper_yaw)

        # Get task value
        grasp_ready_pose = list(get_current_pose(1))
        grasp_ready_pose[0] = robot_x
        grasp_ready_pose[1] = robot_y
        grasp_ready_pose[2] = robot_z

        robot.move_l(grasp_ready_pose)

        # Move to pick
        grasp_pose = grasp_ready_pose.copy()
        grasp_pose[2] = -231

        task_compliance_ctrl(stx=stiffness)

        collision_detected = False

        start_z = int(grasp_ready_pose[2])
        target_z = int(grasp_pose[2])

        for current_z in range(start_z - 5, target_z - 1, -5):
            step_pose = grasp_ready_pose.copy()
            step_pose[2] = float(current_z)

            amovel(step_pose, vel=10, acc=10)

            t_start = time.time()
            while time.time() - t_start < 0.2:
                if check_force_condition(axis=DR_AXIS_Z, min=8, ref=DR_TOOL) == 0:
                    print(f"Collision Detection!!!")
                    collision_detected = True
                    break

            if collision_detected:
                break

        # Pass grasping task
        if collision_detected:
            robot.move_l(grasp_ready_pose)
            release_compliance_ctrl()

        # Lifting cube
        else:
            release_compliance_ctrl()
            robot.grasp()
            robot.move_l(grasp_ready_pose)

        # Place cube
        place_pose = grasp_ready_pose.copy()
        place_pose[0] = -425
        place_pose[1] = -300

        robot.move_l(place_pose)
        robot.release()

    # Done with the task
    robot.move_j(task_home_joint)

if __name__ == '__main__':
    main()

