# UR5

The repo is robot repo for control UR5 with robotiq gripper on noetic.

UR5 repo including:
* robotiq
* universal_robot
* ur_modern_driver
* arm_operation
* ur5_bringup

## ROS Distro Support

|         | Melodic | Noetic  |
|:-------:|:-------:|:-------:|
| branch | [`melodic`](https://github.com/kuolunwang/UR5/tree/melodic) | [`noetic`](https://github.com/kuolunwang/UR5/tree/noetic) |
| Status | supported | supported |

## Clone repo

```
    git clone --recursive git@github.com:kuolunwang/UR5.git
```

### Hardware Setup

1. Make sure you connect UR5 already, and you can ping ip of ur5 (192.168.50.11).
2. Connect robotiq gripper already, and make sure its LED is powerd.

### Software Setup

1. You need to launch ur5_control.launch to connect ur5 and robotiq, and excute control commands.
    ```
        roslaunch ur5_bringup ur5_control.launch
    ```

## Service List

---

| Service Name | Service Type | Service Description |
|:--------:|:--------:|:--------:|
| /ur5/go_home | [Trigger](http://docs.ros.org/en/melodic/api/std_srvs/html/srv/Trigger.html) | Set ur5 to specific home position |
| /robotiq/close_gripper | [Empty](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Set robotiq gripper close |
| /robotiq/open_gripper | [Empty](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Set robotiq gripper open |
| /robotiq/check_grasped | [Trigger](http://docs.ros.org/en/melodic/api/std_srvs/html/srv/Trigger.html) | Check if the gripper is grasped object |
| /robotiq/initial_gripper | [Empty](http://docs.ros.org/en/api/std_srvs/html/srv/Empty.html) | Set robotiq gripper close and open in the begining |
