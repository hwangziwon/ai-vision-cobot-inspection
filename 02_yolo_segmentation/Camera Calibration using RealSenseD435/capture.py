import cv2

from camera import RealSenseD435

def main():
    camera = RealSenseD435(color_resolution=720, depth_mode="720P")
    color_img, depth_img = camera.get_image()

    # printing method 1
    print(f"_color_intrinsics_mat\n{camera._color_intrinsics_mat}")
    print(f"_depth_intrinsics_mat\n{camera._depth_intrinsics_mat}")
    #########

    # printing method 2
    color_intrinsics_mat, depth_intrinsics_mat = camera._init_intrinsics()

    print(f"_color_intrinsics_mat\n{color_intrinsics_mat}")
    print(f"_depth_intrinsics_mat\n{depth_intrinsics_mat}")
    ##########

    cv2.imshow("Original Color image", color_img)
    cv2.imwrite('/home/sht/ros2_ws/src/doosan-robot2/output.png', color_img) # need to edit (check your path)
   
    print(f"Depth image's shape: {depth_img.shape}")
    print(f"Depth image's Y: {depth_img.shape[0]}")
    print(f"Depth image's X: {depth_img.shape[1]}")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()