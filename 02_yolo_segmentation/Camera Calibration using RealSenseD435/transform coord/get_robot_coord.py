import sys
import numpy as np

from camera import RealSenseD435

extrinsic_matrix = np.array([
    [-9.842804480632064701e-01, 9.256802314739166759e-02, -1.504099752379392252e-01, -4.134985284882139456e-01],
    [1.695416795147855760e-01, 7.338012390910285676e-01, -6.578688018258492809e-01, 3.353550706617078325e-01],
    [4.947341172588870517e-02, -6.730281588455686581e-01, -7.379603505156259180e-01, 2.408146706989155172e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
], dtype=np.float32)


def main(args=None):
    camera = RealSenseD435(color_resolution=720, depth_mode="720P")

    # get image
    color_image, depth_image = camera.get_image()

    center_x = 743  # pixel
    center_y = 457  # pixel

    cam_z = depth_image[center_y, center_x]

    yaw = 0.0

    cam_x = np.multiply(center_x - camera._color_intrinsics_mat[0][2], cam_z / 
    camera._color_intrinsics_mat[0][0])

    cam_y = np.multiply(center_y - camera._color_intrinsics_mat[1][2], cam_z / 
    camera._color_intrinsics_mat[1][1])

    cam_coordinate = [cam_x, cam_y, cam_z, yaw]
    print(f"cam_coord\n{cam_coordinate}")

    world_coordinate = extrinsic_matrix[0:3, 0:3] @ cam_coordinate[0:3]
    world_coordinate += extrinsic_matrix[0:3, 3:].flatten()

    print(f"world_coord\n{world_coordinate}")


if __name__ == "__main__":
    main()