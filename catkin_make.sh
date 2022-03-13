#! /bin/bash

catkin_make --pkg realsense2_camera -C ./catkin_ws

touch ./catkin_ws/src/robot/vx300s/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/CATKIN_IGNORE

# python3 cv_bridge
catkin_make --pkg vision_opencv -C ./catkin_ws \
	-DCMAKE_BUILD_TYPE=Release \
	-DPYTHON_EXECUTABLE=/usr/bin/python3 \
	-DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
	-DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so

catkin_make --pkg geometry2 -C ./catkin_ws \
	--cmake-args \
	-DCMAKE_BUILD_TYPE=Release \
	-DPYTHON_EXECUTABLE=/usr/bin/python3 \
	-DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
	-DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin_make -C ./catkin_ws
