# Aruco Pose Estimation with ROS2, using RGB and Depth camera images from Realsense D435

This package allows to use cameras to detect Aruco markers and estimate their poses. It allows to use any camera with ROS2 drivers.
The code is a ROS2 publisher-subscriber working with RGB camera images for marker detection and RGB or depth images for pose estimation. 
It also allows using multiple aruco markers at the same time, and each of them will be published as a separate pose. 
It can also optionally fuse multiple markers that belong to one rigid target into a single board pose for lower jitter and lower pose error.
The code supports many different Aruco dictionaries and sizes of markers.

This package was tested for the Realsense D435 camera, compatible with ROS2 `realsense-ros` driver,
available at [ros2_intel_realsense](https://github.com/IntelRealSense/realsense-ros). The code should work equally well on other and
different cameras, provided a proper calibration of the camera parameters.

## Installation

### Install RealSense ROS2 Wrapper (Required for RealSense cameras)

If using Intel RealSense cameras, install the RealSense ROS2 packages:

```bash
# For ROS2 Humble:
$ sudo apt install ros-humble-librealsense2*
$ sudo apt install ros-humble-realsense2-*

# For ROS2 Iron:
$ sudo apt install ros-iron-librealsense2*
$ sudo apt install ros-iron-realsense2-*
```

### Install Python Dependencies

This package depends on a recent version of OpenCV python library and transforms libraries:

```bash
$ pip3 install -r requirements.txt

$ sudo apt install ros-humble-tf-transformations
# OR for ROS2 Iron:
$ sudo apt install ros-iron-tf-transformations
```

**Note for Conda Users:** If using conda environments with ROS2 Humble, you need to install additional dependencies:
```bash
$ pip uninstall em  # Remove conflicting 'em' package if present
$ pip install catkin_pkg lark empy==3.3.4
$ pip install "numpy<2.0.0"  # ROS2 Humble's cv_bridge requires NumPy 1.x
```

Build the package from source with `colcon build --symlink-install` in the workspace root:

```bash
$ cd /path/to/your/ros2_workspace
$ colcon build --symlink-install
$ source install/setup.bash
```

**Important:** You must source the workspace in every new terminal before using the package:
```bash
$ source install/setup.bash
```

Or add it to your `.bashrc` for automatic sourcing:
```bash
$ echo "source ~/path/to/your/ros2_workspace/install/setup.bash" >> ~/.bashrc
```

## Aruco Pose Detection and Estimation ROS2 nodes description

This node subscribes to the RGB and optionally Depth images from the camera, and the camera inof topic for
intrinsic and distortion parameters. It detects Aruco markers in the RGB image, and estimates their poses using the
camera intrinsic parameters or the depth image. The poses are published as PoseArray message, and the detected markers
are published as ArucoMarkers messages. The output image contains the detected markers and aruco bounding boxes drawn on it.

__Subscribed topics__ (topic names can be changed in the `config/aruco_parameters.yaml` file):

* `/camera/image_raw`: RGB image input (`sensor_msgs.msg.Image`)
* `/camera/depth/image_rect_raw`: Depth image input (`sensor_msgs.msg.Image`)
* `/camera/camera_info`: Camera intrinsic, projection, distortion parameters (`sensor_msgs.msg.CameraInfo`)

__Published topics__ (topic names can be changed in the `config/aruco_parameters.yaml` file):

* `/aruco/poses`: Poses of all detected markers, suitable for rviz visualization - (`geometry_msgs.msg.PoseArray`) - 
* `/aruco/markers`: Provides an array of all poses along with the corresponding marker ids - (`aruco_interfaces.msg.ArucoMarkers`)
* `/aruco/image`: Output image with detected markers drawn on it, for visualization purposes - (`sensor_msgs.msg.Image`)
* `/aruco/board_pose`: Optional fused rigid-board pose estimated from multiple configured markers - (`geometry_msgs.msg.PoseStamped`)

__Parameters__ for the node can be set in the `config/aruco_parameters.yaml` file, and include the following options:

* `marker_size` - size of the markers in meters
* `aruco_dictionary_id` - dictionary type that was used to generate markers (example `DICT_5X5_250`)
* `enable_board_fusion` - enable fused rigid-board pose estimation from multiple configured markers
* `board_config_file` - path to the rigid board yaml layout file
* `board_pose_topic` - topic to publish the fused rigid-board pose
* `board_min_markers` - minimum number of visible configured board markers before the fused pose is published
* `board_pose_refine` - refine the fused pose estimate after RANSAC when OpenCV supports it
* `use_depth_input` - use depth image for pose estimation (default `false`)
* `image_topic` - RGB image topic to subscribe to, provided by the camera ROS2 driver
* `depth_image_topic` - Depth image topic to subscribe to, provided by the camera ROS2 driver
* `camera_info_topic` - Camera info topic to subscribe to, providing intrinsic and distortion parameters
* `camera_frame` - Camera optical frame to use (default to the frame id provided by the camera info message.)
* `detecter_markers_topic` - Topic to publish the detected markers as ArucoMarkers message
* `markers_visualization_topic` - Topic to publish the detected markers as PoseArray message
* `output_image_topic` - Topic to publish the output image with detected markers drawn on it, for visualization purposes

## Running Marker Detection for Pose Estimation

Launch the aruco pose estimation node with this command. The parameters will be loaded from _aruco\_parameters.yaml_,
but can also be changed directly in the launch file with command line arguments.

**Basic launch (camera must be running separately):**

```bash
ros2 launch aruco_pose_estimation aruco_pose_estimation.launch.py
```

**Launch with automatic camera start:**

```bash
ros2 launch aruco_pose_estimation aruco_pose_estimation.launch.py launch_camera:=true
```

**If running camera separately (recommended for better performance):**

```bash
# Terminal 1 - Start camera
ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=640x480x15

# Terminal 2 - Start ArUco detection
ros2 launch aruco_pose_estimation aruco_pose_estimation.launch.py
```

**Change parameters directly in the launch file:**

```bash
ros2 launch aruco_pose_estimation aruco_pose_estimation.launch.py marker_size:=0.05 aruco_dictionary_id:=DICT_5X5_250 launch_camera:=true
```

## Optional Rigid Multi-Marker Board Fusion

If several markers belong to one rigid target, you can keep the existing per-marker outputs and also publish one fused board pose.
The fused pose is estimated from all visible configured board markers in one PnP solve.

### Board layout file

Create a yaml file or start from the sample config at `aruco_pose_estimation/config/aruco_board_layout.yaml`:

```yaml
aruco_board:
  frame_name: board
  markers:
    - id: 0
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
    - id: 1
      xyz: [0.08, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
```

Each `xyz` and `rpy` entry defines the pose of a marker center in the rigid board frame.
The node derives each marker's 3D corners from this pose and the existing global `marker_size`.

### Launch board fusion  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

```bash
ros2 launch aruco_pose_estimation aruco_pose_estimation.launch.py \
  launch_camera:=true \
  enable_board_fusion:=true \
  board_config_file:=config/aruco_board_layout.yaml
```

This publishes:

* the current per-marker outputs on `/aruco/markers` and `/aruco/poses`
* the fused rigid-board pose on `/aruco/board_pose`

### Current limitation

Board fusion is RGB-based in this first version.
If `use_depth_input:=true`, the existing per-marker depth behavior is preserved, but the fused board pose is still estimated from RGB image observations and camera intrinsics.

## Troubleshooting

### No markers detected or nothing showing up

1. **Check if topics are publishing:**
   ```bash
   ros2 topic list
   ```
   You should see `/aruco/markers`, `/aruco/poses`, `/aruco/image`, and camera topics.

2. **Check if markers are being detected:**
   ```bash
   ros2 topic echo /aruco/markers
   ```
   If empty, no markers are detected.

3. **Check the fused board pose when board fusion is enabled:**
   ```bash
   ros2 topic echo /aruco/board_pose
   ```
   If empty, fewer than `board_min_markers` configured board markers are visible, or the board config does not match the visible marker IDs.

4. **View the camera output with detected markers:**
   ```bash
   ros2 run rqt_image_view rqt_image_view
   ```
   Select `/aruco/image` to see the annotated camera feed with detected markers and board-fusion status.

5. **View raw camera image:**
   ```bash
   ros2 run rqt_image_view rqt_image_view
   ```
   Select `/camera/color/image_raw` to verify camera is working.

6. **Verify ArUco marker configuration:**
   - Ensure your printed markers match the dictionary in the config file (default: `DICT_4X4_50`)
   - Verify the `marker_size` parameter matches your physical marker size in meters (default: 0.2m = 20cm)
   - Update [aruco_parameters.yaml](aruco_pose_estimation/config/aruco_parameters.yaml) if needed

7. **Verify board fusion configuration when enabled:**
   - Ensure every rigid board marker ID is present in your `board_config_file`
   - Ensure each marker `xyz` and `rpy` is defined in the same rigid board frame
   - Ensure at least `board_min_markers` configured board markers are visible in the camera

8. **Generate matching ArUco markers:**
   - Use [this online generator](http://chev.me/arucogen/) or OpenCV
   - Select the correct dictionary (e.g., 4x4_50)
   - Print markers at the size specified in your config

### Camera not starting

If the RealSense camera doesn't start automatically, launch it manually:

```bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true
```

Check if camera is detected:
```bash
rs-enumerate-devices
```

## Future updates

It will soon be possible to load the camera calibrated parameters from a yaml configuration file, so that
the camera intrinsic and distortion parameters can be loaded without relying on the camera_info topic or service.
