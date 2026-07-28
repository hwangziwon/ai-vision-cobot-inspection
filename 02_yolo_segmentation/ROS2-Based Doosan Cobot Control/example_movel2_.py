import rclpy
import DR_init
import sys

def main(args=None):
    rclpy.init(args=args)

    ROBOT_ID = "dsr02"
    ROBOT_MODEL = "m0609"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = rclpy.create_node('example_py', namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movel, set_robot_mode, ROBOT_MODE_AUTONOMOUS

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    point_1 = [400, 100, 400, 0, -180, 0]
    point_2 = [600, 100, 400, 0, -180, 0]
    point_3 = [600, -100, 400, 0, -180, 0]
    point_4 = [400, -100, 400, 0, -180, 0]

    for i in range(3):
        movel(point_1, vel=50, acc=50)
        movel(point_2, vel=50, acc=50)
        movel(point_3, vel=50, acc=50)
        movel(point_4, vel=50, acc=50)
        print("#@")

    print("Example complete")
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()