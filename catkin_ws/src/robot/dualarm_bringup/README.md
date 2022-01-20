# Dualarm Bringup

The package is for control dual arm vx300s.

## Hardware Setup

1. Setup dualarm environment.
2. Check your computer can detect /dev/ttyUSB0 and /dev/ttyUSB1

## Software Setup

1. First you need to launch dualarm_control.launch.
\
    It will connect two vx300s and run dualarm_control_node to control vx300s.
    ```
        roslaunch dualarm_bringup dualarm_control.launch
    ```
2. Check rosservice name.
\
    You can see right_arm and left_arm rosservice.
    ```
        rosservice list
    ```

## Service List
\
**In this project the robot_name are right_arm and left_arm, below table will list right_arm service for example, it also use for left_arm.**

---

| Service Name | Service Type | Service Description |
|:--------:|:--------:|:--------:|
| /right_arm/go_home | [Trigger](http://docs.ros.org/en/melodic/api/std_srvs/html/srv/Trigger.html) | Set right vx300s joints to initial value(0) |
|/right_arm/go_sleep| [Trigger](http://docs.ros.org/en/melodic/api/std_srvs/html/srv/Trigger.html) | Set rigth vx300s go back to initial pose |
| /right_arm/go_pose | [ee_pose.srv](https://github.com/kuolunwang/VX300s/blob/main/vx300s_bringup/srv/ee_pose.srv) | Set right vx300s go to specific pose |
| /right_arm/gripper_open| [Trigger](http://docs.ros.org/en/melodic/api/std_srvs/html/srv/Trigger.html) | Set right vx300s end-effector gripper open |
| /right_arm/gripper_close| [Trigger](http://docs.ros.org/en/melodic/api/std_srvs/html/srv/Trigger.html) | Set right vx300s end-effector gripper close |
| /right_arm/check_grasped| [Trigger](http://docs.ros.org/en/melodic/api/std_srvs/html/srv/Trigger.html) | Check if right vx300s end-effector gripper is grasped object