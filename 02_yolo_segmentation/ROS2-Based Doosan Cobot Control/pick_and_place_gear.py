import rclpy
import DR_init
import sys

from robot import Robot

def main(args=None):
    rclpy.init(args=args)

    ROBOT_ID = "dsr02"
    ROBOT_MODEL = "m0609"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = rclpy.create_node('example_py', namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    robot = Robot(node)

    # Define coord value
    gear_pick_point_1 = [391.66, -94.41, 27.91, 80.09, -179.92, 98.24]
    gear_pick_point_2 = [396.34, -202.32, 24.29, 7.03, -179.95, 25.51]
    gear_pick_point_3 = [485.66, -142.00, 25.33, 156.89, -179.97, 174.90]

    gear_pick_up_1 = gear_pick_point_1.copy()
    gear_pick_up_2 = gear_pick_point_2.copy()
    gear_pick_up_3 = gear_pick_point_3.copy()

    gear_pick_up_1[2] += 100.00
    gear_pick_up_2[2] += 100.00
    gear_pick_up_3[2] += 100.00

    gear_place_point_1 = [390.28, 205.85, 24.98, 25.77, 180.00, 43.76]
    gear_place_point_2 = [395.61, 98.48, 26.06, 6.81, -180.00, 25.26]
    gear_place_point_3 = [484.08, 157.98, 25.86, 173.80, -180.00, -167.79]

    gear_place_up_1 = gear_place_point_1.copy()
    gear_place_up_2 = gear_place_point_2.copy()
    gear_place_up_3 = gear_place_point_3.copy()

    gear_place_up_1[2] += 100.00
    gear_place_up_2[2] += 100.00
    gear_place_up_3[2] += 100.00

    # Pick and Place Start!
    robot.home_position()
    robot.release()

    # Gear 1 pick
    robot.move_l(gear_pick_up_1)
    robot.move_l(gear_pick_up_1)
    robot.move_l(gear_pick_point_1)
    robot.grasp()
    robot.move_l(gear_pick_up_1)

    # Gear 1 place
    robot.move_l(gear_place_up_1)
    robot.move_l(gear_place_up_1)
    robot.move_l(gear_place_point_1)
    robot.release()
    robot.move_l(gear_place_up_1)

    # Gear 2 pick
    robot.move_l(gear_pick_up_2)
    robot.move_l(gear_pick_up_2)
    robot.move_l(gear_pick_point_2)
    robot.grasp()
    robot.move_l(gear_pick_up_2)

    # Gear 2 place
    robot.move_l(gear_place_up_2)
    robot.move_l(gear_place_up_2)
    robot.move_l(gear_place_point_2)
    robot.release()
    robot.move_l(gear_place_up_2)

    # Gear 3 pick
    robot.move_l(gear_pick_up_3)
    robot.move_l(gear_pick_up_3)
    robot.move_l(gear_pick_point_3)
    robot.grasp()
    robot.move_l(gear_pick_up_3)

    # Gear 3 place
    robot.move_l(gear_place_up_3)
    robot.move_l(gear_place_up_3)
    robot.move_l(gear_place_point_3)
    robot.release()
    robot.move_l(gear_place_up_3)

    # Pick and Place End!
    robot.home_position()

    print("Example complete")
    rclpy.shutdown()

if __name__ == '__main__':
    main()