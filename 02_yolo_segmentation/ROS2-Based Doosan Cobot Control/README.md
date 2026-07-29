# Doosan Cobot Control with ROS 2

This repository provides a basic guide for setting up the development environment and practicing robot control using the official Python-based ROS 2 communication package (`doosan-robot2`) provided by Doosan Robotics in an Ubuntu 22.04 environment.

---

## 1. Environment Setup

To ensure stable operation of the ROS 2 system, prepare a Linux environment based on Windows dual boot and install the required acceleration libraries.

- **Operating System (OS)**: Shrink the Windows disk volume and create a new partition to install **Ubuntu 22.04 LTS** in a dual-boot configuration. Select **Minimal Installation** during installation for efficient system management.
- **GPU Acceleration and Libraries**: Install **NVIDIA Graphics Driver (v535)**, **CUDA Toolkit 11.8**, and **cuDNN 8.6.0** for future integration with computer vision and AI algorithms. Configure the required environment variables in `~/.bashrc`.
- **ROS 2 Humble Installation**: Install **ROS 2 Humble** by building from the remote source, installing the required packages, and completing the system configuration.

---

## 2. Installing and Running the Doosan Robotics Package

- **Docker Setup**: Install Docker Engine on Ubuntu in advance to provide an isolated environment for deployment.
- **Driver Source Build**: Clone the official Doosan Robotics GitHub repository into the ROS 2 workspace (`~/ros2_ws/src`) using `git clone -b humble`, install the required dependencies with `rosdep`, and compile the source using `colcon build`.
- **Launching the Robot Driver**: After completing the environment setup, run the following launch command to activate the servo motors and establish communication with the physical robot (M0609 model). Modify the robot ID and IP address according to your setup.

```bash
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=192.168.137.128 port:=12345 model:=m0609 name:=dsr02
```

---

## 3. Robot Configuration

Before running the control programs, configure the following settings in the Doosan Workcell Manager to protect the hardware and ensure accurate end-effector calculations.

- **Tool Weight Configuration**: Measure and register the weight of the attached tool, such as a gripper, to prevent motor overload.
- **Tool Center Point (TCP) Configuration**: Define the Tool Center Point (TCP) coordinates to establish the reference for accurate kinematic calculations.

---

## 4. Python API Examples

The provided example scripts demonstrate linear motion control, joint control, gripper control through digital I/O, and code modularization.

### I. Linear Motion Control ([example_movel.py](example_movel.py))

- **Objective**: Move the robot smoothly along a straight path in Task Space using the specified velocity and acceleration.
- **Scenario**: Retrieve the robot's current pose using `get_current_pose()`, then continuously perform a vertical reciprocating motion along the Z-axis over the specified travel distance.

### II. Multi-Point Trajectory Control ([example_movel2.py](example_movel2.py))

- **Objective**: Control an advanced trajectory by sequentially connecting multiple target positions in Cartesian space.
- **Scenario**: Define four task positions (Point 1–Point 4) as the corners of a square, then use a `for` loop to make the end-effector trace a **200 mm × 200 mm square path three times**.

### III. Home Position Control in Joint Space ([home_pose.py](home_pose.py))

- **Objective**: Avoid singularities in Cartesian space and safely return the robot to its home position by directly controlling each joint angle.
- **Scenario**: Define the home joint array (`home_joint = [-180.00, 0.00, 90.00, 0.00, 90.00, 60.00]`) and call the `movej()` function to move the robot smoothly back to the predefined home position.

### IV. Gripper Control Using Digital Output ([gripper_control.py](gripper_control.py))

- **Objective**: Control the grasp and release actions of the end-effector by sending trigger signals through the robot controller's digital output ports.
- **Scenario**: Define the `set_tool_digital_output(index, val)` function for the flange plug signal and combine digital output on/off switching with `time.sleep(1)` delays to implement a repetitive gripper open-and-close sequence.

### V. Robot Control Class Design for Code Modularization ([robot.py](robot.py))

- **Objective**: Integrate the individual control functions developed above (MoveL, MoveJ, Home, and Gripper) into a single Python class to improve code reusability and readability.
- **Structural Advantage**: The `Robot` class creates a ROS 2 node (`node`) internally and dynamically imports the Doosan API library during instance initialization (`__init__`). This allows robot operations such as `self.move_l()`, `self.grasp()`, and `self.release()` to be called and managed through object-oriented methods.
