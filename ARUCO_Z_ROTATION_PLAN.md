# ArUco Z-Axis Rotation Detection (ROS 2 + RealSense)

## Objective
Detect an ArUco marker using a RealSense camera in ROS 2 and compute the rotation of the marker around the Z-axis (yaw), assuming the marker is planar and facing the camera.

## System Overview

RealSense Camera
- /camera/color/image_raw
- /camera/color/camera_info

→ ArUco Detection Node  
→ Pose Estimation (rvec, tvec)  
→ Z-axis Rotation Extraction  
→ Publish / TF / Log

## Dependencies
- realsense2_camera
- rclpy
- sensor_msgs
- geometry_msgs
- tf2_ros
- cv_bridge
- opencv-python
- numpy

## Coordinate Frames

Camera Optical Frame:
- X: right
- Y: down
- Z: forward

Marker Frame:
- Z: normal to marker
- X/Y: along edges

## Implementation Steps

### 1. Create ROS 2 Package
ros2 pkg create aruco_z_rotation --build-type ament_python

### 2. Subscribe to Camera Topics
- /camera/color/image_raw
- /camera/color/camera_info

Extract camera matrix and distortion coefficients.

### 3. Detect ArUco Marker
Use OpenCV:
- cv2.aruco.detectMarkers
- Filter by marker ID

### 4. Estimate Marker Pose
cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, K, D)

Returns:
- rvec
- tvec

### 5. Rotation Matrix
cv2.Rodrigues(rvec) → R (3x3)

### 6. Z-Axis Rotation (Yaw)
yaw = atan2(R[1,0], R[0,0])

yaw_deg = yaw * 180 / pi

### 7. Publish Output
- std_msgs/Float32
- TF: camera_color_optical_frame → aruco_marker_id

### 8. Visualization
- cv2.aruco.drawDetectedMarkers
- cv2.aruco.drawAxis

## Known Pitfalls
- Camera optical frame confusion
- Marker tilt affects yaw
- Wrong marker size → wrong pose

## Future Work
- Tilt compensation
- Filtering
- Multi-marker support
