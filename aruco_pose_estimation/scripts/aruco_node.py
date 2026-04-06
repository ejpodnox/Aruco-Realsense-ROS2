#!/usr/bin/python3
"""
ROS2 wrapper code taken from:
https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco/tree/main

This node locates Aruco AR markers in images and publishes their ids and poses.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
       Pose of all detected markers (suitable for rviz visualization)

    /aruco_markers (aruco_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

    /aruco_image (sensor_msgs.msg.Image)
       Annotated image with marker locations and ids, with markers drawn on it

Parameters:
    marker_size - size of the markers in meters (default .065)
    aruco_dictionary_id - dictionary that was used to generate markers (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/color/image_raw)
    camera_info_topic - camera info topic to subscribe to (default /camera/camera_info)
    camera_frame - camera optical frame to use (default "camera_depth_optical_frame")
    detected_markers_topic - topic to publish detected markers (default /aruco_markers)
    markers_visualization_topic - topic to publish markers visualization (default /aruco_poses)
    output_image_topic - topic to publish annotated image (default /aruco_image)

Author: Simone Giampà
Version: 2024-01-29

"""

# ROS2 imports
import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import message_filters

# Python imports
import numpy as np
import cv2
import os

# Local imports for custom defined functions
from aruco_pose_estimation.board_fusion import load_board_configuration
from aruco_pose_estimation.utils import ARUCO_DICT
from aruco_pose_estimation.pose_estimation import pose_estimation

# ROS2 message imports
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, PoseStamped
from aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")

        self.initialize_parameters()

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(self.dictionary_id_name)
            # check if the dictionary_id is a valid dictionary inside ARUCO_DICT values
            if dictionary_id not in ARUCO_DICT.values():
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(self.dictionary_id_name)
            )
            options = "\n".join([s for s in ARUCO_DICT])
            self.get_logger().error("valid options: {}".format(options))

        # Set up subscriptions to the camera info and camera image topics

        # camera info topic for the camera calibration parameters
        self.info_sub = self.create_subscription(
            CameraInfo, self.info_topic, self.info_callback, qos_profile_sensor_data
        )

        # select the type of input to use for the pose estimation
        if (bool(self.use_depth_input)):
            # use both rgb and depth image topics for the pose estimation

            # create a message filter to synchronize the image and depth image topics
            self.image_sub = message_filters.Subscriber(self, Image, self.image_topic,
                                                        qos_profile=qos_profile_sensor_data)
            self.depth_image_sub = message_filters.Subscriber(self, Image, self.depth_image_topic,
                                                              qos_profile=qos_profile_sensor_data)

            # create synchronizer between the 2 topics using message filters and approximate time policy
            # slop is the maximum time difference between messages that are considered synchronized
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_image_sub], queue_size=10, slop=0.05
            )
            self.synchronizer.registerCallback(self.rgb_depth_sync_callback)
        else:
            # rely only on the rgb image topic for the pose estimation

            # create a subscription to the image topic
            self.image_sub = self.create_subscription(
                Image, self.image_topic, self.image_callback, qos_profile_sensor_data
            )

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, self.markers_visualization_topic, 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, self.detected_markers_topic, 10)
        self.image_pub = self.create_publisher(Image, self.output_image_topic, 10)
        self.board_pose_pub = None

        if self.enable_board_fusion:
            try:
                board_config_path = self.resolve_board_config_path(self.board_config_file)
                self.board_config = load_board_configuration(board_config_path, self.marker_size)
            except ValueError as exc:
                raise RuntimeError(f"Invalid board fusion configuration: {exc}") from exc

            self.board_pose_pub = self.create_publisher(PoseStamped, self.board_pose_topic, 10)
            self.get_logger().info(
                "Board fusion enabled for frame '%s' with markers %s" % (
                    self.board_config.frame_name,
                    sorted(self.board_config.markers.keys()),
                )
            )
        else:
            self.board_config = None

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        # Build detector in a way that works across OpenCV ArUco API versions.
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        else:
            self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)

        if hasattr(cv2.aruco, "ArucoDetector") and hasattr(cv2.aruco, "DetectorParameters"):
            self.aruco_parameters = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
            self.get_logger().info("Using OpenCV ArUcoDetector API.")
        else:
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None
            self.get_logger().info("Using legacy OpenCV ArUco detectMarkers API.")

        self.bridge = CvBridge()

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        # get the intrinsic matrix and distortion coefficients from the camera info
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)

        self.get_logger().info("Camera info received.")
        self.get_logger().info("Intrinsic matrix: {}".format(self.intrinsic_mat))
        self.get_logger().info("Distortion coefficients: {}".format(self.distortion))
        self.get_logger().info("Camera frame: {}x{}".format(self.info_msg.width, self.info_msg.height))

        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg: Image):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        # convert the image messages to cv2 format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        self.process_frame(cv_image, None, img_msg.header.stamp)

    def depth_image_callback(self, depth_msg: Image):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

    def rgb_depth_sync_callback(self, rgb_msg: Image, depth_msg: Image):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        # convert the image messages to cv2 format
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        self.process_frame(cv_image, cv_depth_image, rgb_msg.header.stamp)

    def process_frame(self, rgb_image: np.ndarray, depth_image: np.ndarray, stamp):
        markers, pose_array = self.create_marker_messages(stamp)

        frame, pose_array, markers, board_status = pose_estimation(
            rgb_frame=rgb_image,
            depth_frame=depth_image,
            aruco_detector=self.aruco_detector,
            marker_size=self.marker_size,
            matrix_coefficients=self.intrinsic_mat,
            distortion_coefficients=self.distortion,
            pose_array=pose_array,
            markers=markers,
            aruco_dictionary=self.aruco_dictionary,
            aruco_parameters=self.aruco_parameters,
            board_config=self.board_config,
            board_min_markers=self.board_min_markers,
            board_pose_refine=self.board_pose_refine,
        )

        if len(markers.marker_ids) > 0:
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        if self.board_pose_pub is not None and board_status is not None and board_status.pose_estimate is not None:
            self.board_pose_pub.publish(self.create_board_pose_message(stamp, board_status.pose_estimate))

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

    def create_marker_messages(self, stamp):
        markers = ArucoMarkers()
        pose_array = PoseArray()

        frame_id = self.resolve_output_frame_id()
        markers.header.frame_id = frame_id
        pose_array.header.frame_id = frame_id
        markers.header.stamp = stamp
        pose_array.header.stamp = stamp
        return markers, pose_array

    def create_board_pose_message(self, stamp, pose_estimate):
        message = PoseStamped()
        message.header.frame_id = self.resolve_output_frame_id()
        message.header.stamp = stamp
        message.pose.position.x = float(pose_estimate.translation[0])
        message.pose.position.y = float(pose_estimate.translation[1])
        message.pose.position.z = float(pose_estimate.translation[2])
        message.pose.orientation.x = float(pose_estimate.quaternion[0])
        message.pose.orientation.y = float(pose_estimate.quaternion[1])
        message.pose.orientation.z = float(pose_estimate.quaternion[2])
        message.pose.orientation.w = float(pose_estimate.quaternion[3])
        return message

    def resolve_output_frame_id(self):
        if self.camera_frame == "":
            return self.info_msg.header.frame_id
        return self.camera_frame

    def resolve_board_config_path(self, config_path: str) -> str:
        expanded_path = os.path.expanduser(config_path.strip())
        if os.path.isabs(expanded_path):
            return expanded_path

        if os.path.exists(expanded_path):
            return os.path.abspath(expanded_path)

        package_relative_path = os.path.join(
            get_package_share_directory("aruco_pose_estimation"),
            expanded_path,
        )
        return package_relative_path

    def initialize_parameters(self):
        # Declare and read parameters from aruco_params.yaml
        self.declare_parameter(
            name="marker_size",
            value=0.0625,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )

        self.declare_parameter(
            name="use_depth_input",
            value=True,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Use depth camera input for pose estimation instead of RGB image",
            ),
        )

        self.declare_parameter(
            name="image_topic",
            value="/camera/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="depth_image_topic",
            value="/camera/depth/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Depth camera topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )

        self.declare_parameter(
            name="detected_markers_topic",
            value="/aruco_markers",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish detected markers as array of marker ids and poses",
            ),
        )

        self.declare_parameter(
            name="markers_visualization_topic",
            value="/aruco_poses",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish markers as pose array",
            ),
        )

        self.declare_parameter(
            name="output_image_topic",
            value="/aruco_image",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish annotated images with markers drawn on them",
            ),
        )

        self.declare_parameter(
            name="enable_board_fusion",
            value=False,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Enable fused rigid-board pose estimation from multiple configured markers.",
            ),
        )

        self.declare_parameter(
            name="board_config_file",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Path to the rigid board marker layout YAML file.",
            ),
        )

        self.declare_parameter(
            name="board_pose_topic",
            value="/aruco/board_pose",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish the fused rigid-board pose.",
            ),
        )

        self.declare_parameter(
            name="board_min_markers",
            value=2,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Minimum number of configured markers required before publishing a fused board pose.",
            ),
        )

        self.declare_parameter(
            name="board_pose_refine",
            value=True,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Refine the fused board pose after RANSAC using Levenberg-Marquardt when OpenCV supports it.",
            ),
        )

        # read parameters from aruco_params.yaml and store them
        self.marker_size = (
            self.get_parameter("marker_size").get_parameter_value().double_value
        )
        self.get_logger().info(f"Marker size: {self.marker_size}")

        self.dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        )
        self.get_logger().info(f"Marker type: {self.dictionary_id_name}")

        self.use_depth_input = (
            self.get_parameter("use_depth_input").get_parameter_value().bool_value
        )
        self.get_logger().info(f"Use depth input: {self.use_depth_input}")

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Input image topic: {self.image_topic}")

        self.depth_image_topic = (
            self.get_parameter("depth_image_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Input depth image topic: {self.depth_image_topic}")

        self.info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image camera info topic: {self.info_topic}")

        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self.get_logger().info(f"Camera frame: {self.camera_frame}")

        # Output topics
        self.detected_markers_topic = (
            self.get_parameter("detected_markers_topic").get_parameter_value().string_value
        )

        self.markers_visualization_topic = (
            self.get_parameter("markers_visualization_topic").get_parameter_value().string_value
        )

        self.output_image_topic = (
            self.get_parameter("output_image_topic").get_parameter_value().string_value
        )

        self.enable_board_fusion = (
            self.get_parameter("enable_board_fusion").get_parameter_value().bool_value
        )
        self.get_logger().info(f"Enable board fusion: {self.enable_board_fusion}")

        self.board_config_file = (
            self.get_parameter("board_config_file").get_parameter_value().string_value
        )
        self.get_logger().info(f"Board config file: {self.board_config_file}")

        self.board_pose_topic = (
            self.get_parameter("board_pose_topic").get_parameter_value().string_value
        )

        self.board_min_markers = (
            self.get_parameter("board_min_markers").get_parameter_value().integer_value
        )
        self.get_logger().info(f"Board minimum visible markers: {self.board_min_markers}")

        self.board_pose_refine = (
            self.get_parameter("board_pose_refine").get_parameter_value().bool_value
        )
        self.get_logger().info(f"Board pose refine: {self.board_pose_refine}")

        if self.enable_board_fusion and self.board_config_file.strip() == "":
            raise RuntimeError(
                "Board fusion is enabled but 'board_config_file' is empty. "
                "Set it to a valid rigid board YAML file."
            )


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
