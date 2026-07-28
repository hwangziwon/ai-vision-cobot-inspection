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

    from DSR_ROBOT2 import movej, set_robot_mode, move_home, ROBOT_MODE_AUTONOMOUS

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    
    #1
    #point_1 = [368.00, 6.00, 424.00, 10.00, -179.00, 10.00]
    #point_2 = [368.00, 6.00, 394.00, 10.00, -179.00, 10.00]

    # movel(point_1, vel=50, acc=50)
    # movel(point_2, vel=50, acc=50)

    #2
    # move_home(1) # packaging home pose
    # move_home(0) # custom home pose
    
    #3
    # task_home_joint = [-180.00, 0.00, 90.00, 0.00, 90.00, 60.00]
    # movej(task_home_joint, vel=25, acc=25)

    print("Example complete")
    rclpy.shutdown()

if __name__ == "__main__":
    main()