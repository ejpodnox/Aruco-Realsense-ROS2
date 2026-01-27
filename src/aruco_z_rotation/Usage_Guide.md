# How to Use Aruco Z Rotator

This guide explains how to build and run the `aruco_z_rotation` package.

## Prerequisites

Ensure you have the following installed:
- ROS 2 Humble (or compatible)
- `realsense2_camera` package
- OpenCV (usually comes with ROS 2)

## 1. Build the Package

From the root of your workspace:

```bash
colcon build --symlink-install --packages-select aruco_z_rotation
source install/setup.bash
```

## 2. Run the RealSense Camera

Start your RealSense camera node first. Ideally, use a terminal for this:

```bash
ros2 launch realsense2_camera rs_launch.py
```

*Note: Ensure the topic names match. The node expects `/camera/color/image_raw` and `/camera/color/camera_info`.*

## 3. Run the Detection Node

Run the node using the launch file:

```bash
ros2 launch aruco_z_rotation aruco_z_rotation.launch.py marker_id:=0 marker_size:=0.05
```

### Parameters
- `marker_id`: The integer ID of the ArUco marker you want to track (default: 0).
- `marker_size`: The physical size of the marker in meters (default: 0.05).

## 4. Visualization & Output

### Terminal Output
The node will log the detected Yaw angle in degrees to the terminal:
```
[INFO] [aruco_z_rotation_node]: Marker 0 Yaw: 45.23 degrees
```

### ROS Topic
Subscribe to the topic to get the float value:
```bash
ros2 topic echo /aruco_z_rotation
```

### TF Visualization (RViz)
1. Open RViz: `ros2 run rviz2 rviz2`
2. Set "Fixed Frame" to `camera_color_optical_frame`.
3. Add "TF" verification.
4. You should see `aruco_marker_0` frame moving relative to the camera.

## Troubleshooting

- **No detection?**
    - Check if the camera is publishing images: `ros2 topic hz /camera/color/image_raw`
    - Verify your marker ID matches the physical marker.
    - Check lighting conditions.
    - Ensure the marker belongs to dictionary `DICT_4X4_50` (or change it in `z_rotation_node.py`).

- **Wrong rotation?**
    - The code assumes a planar marker. Z-rotation corresponds to rotation around the marker's normal vector.

