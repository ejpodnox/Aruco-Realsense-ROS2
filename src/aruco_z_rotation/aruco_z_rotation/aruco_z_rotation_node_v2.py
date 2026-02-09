#!/usr/bin/env python3
"""
ArUco Z-Rotation Tracker for Screw Task Evaluation

Key improvements over basic version:
1. Extracts rotation about MARKER'S Z-axis (normal), not camera Z-axis
2. Synchronized start trigger via service or topic
3. Publishes both raw and accumulated angles
4. Better CSV logging with timestamps for offline sync

Usage:
  ros2 run your_package aruco_z_rotation_node_v2.py \
    --ros-args -p marker_id:=0 -p marker_size:=0.05
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, PointStamped
from std_msgs.msg import Float32, Bool, Empty
from std_srvs.srv import Trigger
import math
import time


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw])


def so3_log(R):
    """Log map SO(3) -> so(3), returns rotation vector."""
    R = np.asarray(R, dtype=float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=float)
    w_hat = (R - R.T) * (0.5 / np.sin(theta))
    return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=float)


def signed_angle_about_axis(R_rel, axis):
    """Extract signed rotation angle about a specific axis from relative rotation."""
    rotvec = so3_log(R_rel)
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return 0.0
    axis = axis / axis_norm
    return float(np.dot(rotvec, axis))


class ArucoZRotationNode(Node):
    def __init__(self):
        super().__init__('aruco_z_rotation_node')

        # Parameters
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size', 0.05)
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('dictionary_id', 'DICT_4X4_50')
        self.declare_parameter('show_window', True)
        self.declare_parameter('csv_output', 'aruco_rotation_data.csv')
        self.declare_parameter('max_delta_deg', 30.0)  # Reject jumps larger than this
        self.declare_parameter('sign_flip', False)  # Flip sign if needed to match robot convention

        self.target_marker_id = self.get_parameter('marker_id').value
        self.marker_size = self.get_parameter('marker_size').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        dictionary_id_str = self.get_parameter('dictionary_id').value
        self.show_window = self.get_parameter('show_window').value
        self.csv_output = self.get_parameter('csv_output').value
        self.max_delta_deg = self.get_parameter('max_delta_deg').value
        self.sign_flip = self.get_parameter('sign_flip').value

        # State
        self.accumulated_angle = 0.0  # Total accumulated rotation (degrees)
        self.R_ref = None  # Reference rotation matrix (set on reset)
        self.R_prev = None  # Previous frame rotation matrix
        self.recording = False  # Only accumulate when recording
        self.frame_count = 0
        
        # CSV
        self.csv_file = None
        if self.csv_output:
            try:
                self.csv_file = open(self.csv_output, 'w')
                self.csv_file.write("t_sec,frame,marker_detected,theta_instant_deg,theta_accumulated_deg\n")
                self.get_logger().info(f"CSV logging to: {self.csv_output}")
            except Exception as e:
                self.get_logger().error(f"Failed to create CSV: {e}")

        # CV Bridge
        self.bridge = CvBridge()

        # ArUco setup
        self._setup_aruco(dictionary_id_str)

        # Camera intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False

        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers
        self.theta_pub = self.create_publisher(Float32, '/aruco/theta_deg', 10)
        self.theta_stamped_pub = self.create_publisher(PointStamped, '/aruco/theta_stamped', 10)
        self.recording_pub = self.create_publisher(Bool, '/aruco/recording', 10)

        # Subscribers
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, self.camera_topic, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Empty, '/aruco/reset', self.reset_callback, 10)
        self.create_subscription(Empty, '/aruco/start', self.start_callback, 10)
        self.create_subscription(Empty, '/aruco/stop', self.stop_callback, 10)

        # Services
        self.reset_srv = self.create_service(Trigger, '/aruco/reset_service', self.reset_service_callback)
        self.start_srv = self.create_service(Trigger, '/aruco/start_service', self.start_service_callback)

        self.get_logger().info(f"ArUco Z-Rotation Node V2 started. Marker ID: {self.target_marker_id}")
        self.get_logger().info("Commands: publish to /aruco/reset, /aruco/start, /aruco/stop")
        self.get_logger().info("Or call services: /aruco/reset_service, /aruco/start_service")

    def _setup_aruco(self, dictionary_id_str):
        """Initialize ArUco detector."""
        # Initialize ArUco Dictionary
        try:
            # Check if dictionary string exists in cv2.aruco
            if hasattr(cv2.aruco, dictionary_id_str):
                dictionary_id = getattr(cv2.aruco, dictionary_id_str)
                self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            elif dictionary_id_str == 'DICT_ORIGINAL': # Handle shorthand
                 self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            else:
                self.get_logger().error(f"Dictionary {dictionary_id_str} not found in cv2.aruco! Defaulting to DICT_5X5_100")
                self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

            self.aruco_parameters = cv2.aruco.DetectorParameters()
            
            # --- Robustness Tuning ---
            # Improves pose accuracy, especially when tilted
            self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 
            # Allow detecting smaller markers (useful when marker is tilted away)
            self.aruco_parameters.minMarkerPerimeterRate = 0.02 
            # Tune adaptive thresholding for better detection in varying light
            self.aruco_parameters.adaptiveThreshWinSizeMin = 3
            self.aruco_parameters.adaptiveThreshWinSizeMax = 30
            self.aruco_parameters.adaptiveThreshWinSizeStep = 10
            # -------------------------

            self.detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
            self.use_new_api = True
        except AttributeError:
             # Fallback for older OpenCV versions
            target_dict_id = cv2.aruco.DICT_ARUCO_ORIGINAL if dictionary_id_str == 'DICT_ORIGINAL' else getattr(cv2.aruco, dictionary_id_str)
            self.aruco_dictionary = cv2.aruco.Dictionary_get(target_dict_id)
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()
            
            # --- Robustness Tuning (Old API) ---
            self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 
            self.aruco_parameters.minMarkerPerimeterRate = 0.02
            self.aruco_parameters.adaptiveThreshWinSizeMin = 3
            self.aruco_parameters.adaptiveThreshWinSizeMax = 30
            self.aruco_parameters.adaptiveThreshWinSizeStep = 10
            # -----------------------------------

            self.use_new_api = False
        # try:
        #     if hasattr(cv2.aruco, dictionary_id_str):
        #         dictionary_id = getattr(cv2.aruco, dictionary_id_str)
        #         self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        #     else:
        #         self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        #     self.aruco_parameters = cv2.aruco.DetectorParameters()
        #     self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        #     self.aruco_parameters.minMarkerPerimeterRate = 0.02
        #     self.aruco_parameters.adaptiveThreshWinSizeMin = 3
        #     self.aruco_parameters.adaptiveThreshWinSizeMax = 30
        #     self.aruco_parameters.adaptiveThreshWinSizeStep = 10

        #     self.detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
        #     self.use_new_api = True
        # except AttributeError:
        #     self.aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        #     self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        #     self.use_new_api = False

    def reset_callback(self, msg):
        self._reset()

    def start_callback(self, msg):
        self._start_recording()

    def stop_callback(self, msg):
        self._stop_recording()

    def reset_service_callback(self, request, response):
        self._reset()
        response.success = True
        response.message = "ArUco accumulator reset"
        return response

    def start_service_callback(self, request, response):
        self._start_recording()
        response.success = True
        response.message = f"Recording started. Accumulated: {self.accumulated_angle:.2f} deg"
        return response

    def _reset(self):
        """Reset accumulator and reference."""
        self.accumulated_angle = 0.0
        self.R_ref = None
        self.R_prev = None
        self.frame_count = 0
        self.get_logger().info("=== ARUCO RESET ===")

    def _start_recording(self):
        """Start accumulating rotation."""
        self.recording = True
        # Reset on start
        self._reset()
        self.get_logger().info("=== ARUCO RECORDING STARTED ===")

    def _stop_recording(self):
        """Stop accumulating rotation."""
        self.recording = False
        self.get_logger().info(f"=== ARUCO RECORDING STOPPED === Total: {self.accumulated_angle:.2f} deg")

    def camera_info_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.got_camera_info = True
            self.get_logger().info("Camera info received.")

    def image_callback(self, msg):
        if not self.got_camera_info:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dictionary, parameters=self.aruco_parameters)

        marker_detected = False
        theta_instant = 0.0

        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.target_marker_id:
                    marker_detected = True

                    # Estimate pose
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], self.marker_size, self.camera_matrix, self.dist_coeffs
                    )

                    # Get rotation matrix (marker frame relative to camera frame)
                    R_marker_camera, _ = cv2.Rodrigues(rvec)

                    # ============================================================
                    # KEY FIX: Extract rotation about MARKER'S Z-axis (normal)
                    # ============================================================
                    # The marker's Z-axis in camera frame is R_marker_camera[:, 2]
                    # We want to track rotation about this axis.
                    
                    marker_z_axis = R_marker_camera[:, 2]  # Marker normal in camera frame

                    if self.R_ref is None:
                        # First detection: set reference
                        self.R_ref = R_marker_camera.copy()
                        self.R_prev = R_marker_camera.copy()
                        theta_instant = 0.0
                    else:
                        # Compute incremental rotation from previous frame
                        # dR = R_prev^T @ R_current
                        dR = self.R_prev.T @ R_marker_camera
                        
                        # Extract rotation about marker Z-axis
                        # Use the reference marker Z-axis for consistency
                        ref_z_axis = self.R_ref[:, 2]
                        
                        dtheta_rad = signed_angle_about_axis(dR, ref_z_axis)
                        dtheta_deg = np.degrees(dtheta_rad)

                        # Apply sign flip if needed
                        if self.sign_flip:
                            dtheta_deg = -dtheta_deg

                        # Reject large jumps (tracking failure)
                        if abs(dtheta_deg) < self.max_delta_deg:
                            if self.recording:
                                self.accumulated_angle += dtheta_deg
                        else:
                            self.get_logger().warn(f"Rejected jump: {dtheta_deg:.1f} deg")

                        # Compute instantaneous angle from reference
                        R_rel = self.R_ref.T @ R_marker_camera
                        theta_instant = np.degrees(signed_angle_about_axis(R_rel, ref_z_axis))
                        if self.sign_flip:
                            theta_instant = -theta_instant

                        self.R_prev = R_marker_camera.copy()

                    # Publish
                    self.theta_pub.publish(Float32(data=float(self.accumulated_angle)))

                    theta_msg = PointStamped()
                    theta_msg.header.stamp = msg.header.stamp
                    theta_msg.header.frame_id = msg.header.frame_id
                    theta_msg.point.x = float(theta_instant)  # Instantaneous angle from reference
                    theta_msg.point.y = float(self.accumulated_angle)  # Accumulated angle
                    theta_msg.point.z = float(marker_detected)  # Detection flag
                    self.theta_stamped_pub.publish(theta_msg)

                    # Broadcast TF
                    self._broadcast_tf(msg.header, marker_id, tvec, R_marker_camera)

                    # Draw visualization
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

                    break  # Only process target marker

        # CSV logging
        self.frame_count += 1
        if self.csv_file:
            t_sec = time.time()
            self.csv_file.write(f"{t_sec},{self.frame_count},{int(marker_detected)},{theta_instant:.4f},{self.accumulated_angle:.4f}\n")
            if self.frame_count % 50 == 0:
                self.csv_file.flush()

        # Publish recording status
        self.recording_pub.publish(Bool(data=self.recording))

        # Display
        if self.show_window:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

            # Status text
            status = "RECORDING" if self.recording else "PAUSED"
            color = (0, 255, 0) if self.recording else (0, 165, 255)
            cv2.putText(cv_image, f"[{status}] ID:{self.target_marker_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(cv_image, f"Accumulated: {self.accumulated_angle:.1f} deg", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(cv_image, f"Instant: {theta_instant:.1f} deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if not marker_detected:
                cv2.putText(cv_image, "MARKER NOT DETECTED", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("ArUco Tracker V2", cv_image)
            cv2.waitKey(1)

    def _broadcast_tf(self, header, marker_id, tvec, R):
        """Broadcast marker TF."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = header.frame_id
        t.child_frame_id = f"aruco_marker_{marker_id}"

        tvec_flat = tvec.flatten()
        t.transform.translation.x = float(tvec_flat[0])
        t.transform.translation.y = float(tvec_flat[1])
        t.transform.translation.z = float(tvec_flat[2])

        q = rotation_matrix_to_quaternion(R)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoZRotationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
