#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped, PointStamped
from std_msgs.msg import Float32
import math

class ArucoZRotationNode(Node):
    def __init__(self):
        super().__init__('aruco_z_rotation_node')

        # Parameters
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size', 0.05)  # meters
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('dictionary_id', 'DICT_4X4_50')
        self.declare_parameter('show_window', True)
        self.declare_parameter('csv_output', 'rotation_data.csv')

        self.target_marker_id = self.get_parameter('marker_id').value
        self.marker_size = self.get_parameter('marker_size').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        dictionary_id_str = self.get_parameter('dictionary_id').value
        self.show_window = self.get_parameter('show_window').value
        self.csv_output = self.get_parameter('csv_output').value

        # Accumulators
        self.accumulated_yaw = 0.0
        self.last_yaw = None
        self.wrap_around_threshold = 300.0 # deg, threshold to detect 359->0 transition

        # Video recording
        self.is_recording = False
        self.video_writer = None
        self.video_filename = None
        self.frame_size = None

        # CSV initialization
        if self.csv_output:
            try:
                self.csv_file = open(self.csv_output, 'w')
                self.csv_file.write("timestamp,marker_id,yaw_deg,accumulated_yaw_deg\n")
            except Exception as e:
                self.get_logger().error(f"Failed to create CSV file: {e}")
                self.csv_output = None

        # Initialize CV Bridge
        self.bridge = CvBridge()

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

        # Camera Intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False

        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publishers
        self.rotation_pub = self.create_publisher(Float32, 'aruco_z_rotation', 10)
        self.rotation_stamped_pub = self.create_publisher(PointStamped, 'aruco_z_rotation_stamped', 10)

        # Subscribers
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, self.camera_topic, self.image_callback, qos_profile_sensor_data)

        self.get_logger().info(f"Aruco Z Rotation Node Started. Tracking ID: {self.target_marker_id}")

    def camera_info_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.got_camera_info = True
            self.get_logger().info("Camera Info received and set.")

    def image_callback(self, msg):
        if not self.got_camera_info:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge convert error: {e}")
            return

        # Detect Markers
        corners, ids, rejected = [], [], []
        
        # Convert to grayscale for detection (often more robust)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_api:
             corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
             corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dictionary, parameters=self.aruco_parameters)

        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.target_marker_id:
                    # Estimate Pose
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.marker_size, self.camera_matrix, self.dist_coeffs)
                    
                    # Compute Z Rotation
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    
                    # Extract Yaw (Z-rotation) from planar marker
                    # Assuming marker is on a X-Y plane, Z is normal.
                    # R = [r00 r01 r02]
                    #     [r10 r11 r12]
                    #     [r20 r21 r22]
                    # yaw = atan2(r10, r00)
                    yaw_rad = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    yaw_deg = math.degrees(yaw_rad)

                    # Normalize yaw to 0-360 for consistent logic if needed, but atan2 returns -180 to 180
                    # Let's keep -180 to 180 as 'current'
                    
                    # Accumulate Rotation
                    if self.last_yaw is not None:
                        delta = yaw_deg - self.last_yaw
                        # Detect wrap-around (e.g. 179 -> -179 is a small step of +2, not -358)
                        if delta < -180.0:
                             delta += 360.0
                        elif delta > 180.0:
                             delta -= 360.0
                        self.accumulated_yaw += delta
                    else:
                        self.accumulated_yaw = 0.0 # Start from 0 relative to first detection or keep absolute?
                        # Usually "accumulated" means relative to start. 
                        # Or if we want total turns, we just integrate delta.
                        # Let's assume we start accumulation from 0 upon first detection.
                        pass
                    
                    self.last_yaw = yaw_deg

                    # Publish Rotation
                    # self.rotation_pub.publish(Float32(data=yaw_deg))
                    # Publish (unstamped, for backward compatibility)
                    self.rotation_pub.publish(Float32(data=float(yaw_deg)))

                    # Publish stamped (for synchronization)
                    m = PointStamped()
                    m.header.stamp = msg.header.stamp  # best: use the image timestamp
                    m.header.frame_id = msg.header.frame_id
                    m.point.x = 0.0
                    m.point.y = 0.0
                    m.point.z = float(self.accumulated_yaw)
                    self.rotation_stamped_pub.publish(m)

                    self.get_logger().info(f"Marker {marker_id} Yaw: {yaw_deg:.2f}, Total: {self.accumulated_yaw:.2f} deg")

                    # Broadcast TF
                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = msg.header.frame_id
                    t.child_frame_id = f"aruco_marker_{marker_id}"
                    
                    # Fix for TypeError: only size-1 arrays can be converted to Python scalars
                    # tvec structure from estimatePoseSingleMarkers is [[[x, y, z]]] (1, 1, 3) or [[x, y, z]] (1, 3) depending on version
                    tvec_flat = tvec.flatten()
                    t.transform.translation.x = float(tvec_flat[0])
                    t.transform.translation.y = float(tvec_flat[1])
                    t.transform.translation.z = float(tvec_flat[2])

                    # Convert rvec to quaternion
                    # Note: Need simpler conversion or use scipy/tf_transformations
                    # For minimal dependency, approximating or doing rodrigues->quat
                    # Let's trust cv2.Rodrigues -> Rotation Matrix -> Quaternion
                    
                    q = self.detect_rotation_matrix_to_quaternion(rotation_matrix)
                    t.transform.rotation.x = q[0]
                    t.transform.rotation.y = q[1]
                    t.transform.rotation.z = q[2]
                    t.transform.rotation.w = q[3]

                    self.tf_broadcaster.sendTransform(t)

                    # Record to CSV
                    if self.csv_output:
                        timestamp = self.get_clock().now().nanoseconds / 1e9
                        self.csv_file.write(f"{timestamp},{marker_id},{yaw_deg:.4f},{self.accumulated_yaw:.4f}\n")
                        self.csv_file.flush()

                    # Draw Axis for debug
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                    
                    # Draw text
                    cv2.putText(cv_image, f"ID: {marker_id} Yaw: {yaw_deg:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(cv_image, f"Total: {self.accumulated_yaw:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        if self.show_window:
            # Draw all detected markers to help debug
            if ids is not None and len(ids) > 0:
                 cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            if ids is not None and len(ids) > 0:
                 for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == self.target_marker_id:
                        # Estimate Pose
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.marker_size, self.camera_matrix, self.dist_coeffs)
                        
                        # Compute Z Rotation
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        
                        # Extract Yaw (Z-rotation) from planar marker
                        # Assuming marker is on a X-Y plane, Z is normal.
                        # R = [r00 r01 r02]
                        #     [r10 r11 r12]
                        #     [r20 r21 r22]
                        # yaw = atan2(r10, r00)
                        yaw_rad = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                        yaw_deg = math.degrees(yaw_rad)

                        # Publish Rotation
                        self.rotation_pub.publish(Float32(data=yaw_deg))
                        self.get_logger().info(f"Marker {marker_id} Yaw: {yaw_deg:.2f} degrees")

                        # Broadcast TF
                        t = TransformStamped()
                        t.header.stamp = self.get_clock().now().to_msg()
                        t.header.frame_id = msg.header.frame_id
                        t.child_frame_id = f"aruco_marker_{marker_id}"
                        
                        # Fix for TypeError: only size-1 arrays can be converted to Python scalars
                        # tvec structure from estimatePoseSingleMarkers is [[[x, y, z]]] (1, 1, 3) or [[x, y, z]] (1, 3) depending on version
                        tvec_flat = tvec.flatten()
                        t.transform.translation.x = float(tvec_flat[0])
                        t.transform.translation.y = float(tvec_flat[1])
                        t.transform.translation.z = float(tvec_flat[2])

                        # Convert rvec to quaternion
                        # Note: Need simpler conversion or use scipy/tf_transformations
                        # For minimal dependency, approximating or doing rodrigues->quat
                        # Let's trust cv2.Rodrigues -> Rotation Matrix -> Quaternion
                        
                        q = self.detect_rotation_matrix_to_quaternion(rotation_matrix)
                        t.transform.rotation.x = q[0]
                        t.transform.rotation.y = q[1]
                        t.transform.rotation.z = q[2]
                        t.transform.rotation.w = q[3]

                        self.tf_broadcaster.sendTransform(t)

                        # Record to CSV
                        if self.csv_output:
                            timestamp = self.get_clock().now().nanoseconds / 1e9
                            self.csv_file.write(f"{timestamp},{marker_id},{yaw_deg:.4f},{self.accumulated_yaw:.4f}\n")
                            self.csv_file.flush()

                        # Draw Axis for debug
                        cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                        
                        # Draw text
                        cv2.putText(cv_image, f"ID: {marker_id} Yaw: {yaw_deg:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(cv_image, f"Total: {self.accumulated_yaw:.1f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        if self.show_window:
            # Show recording indicator if recording
            if self.is_recording:
                cv2.circle(cv_image, (cv_image.shape[1] - 30, 30), 15, (0, 0, 255), -1)  # Red dot
                cv2.putText(cv_image, "REC", (cv_image.shape[1] - 80, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Show instructions
            cv2.putText(cv_image, "R: Record | T: Reset", (10, cv_image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("ArUco Detection", cv_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # 'R' or 'r' to toggle recording
            if key == ord('r') or key == ord('R'):
                if not self.is_recording:
                    # Start recording
                    self.frame_size = (cv_image.shape[1], cv_image.shape[0])
                    timestamp_str = str(int(self.get_clock().now().nanoseconds / 1e6))
                    self.video_filename = f"aruco_recording_{timestamp_str}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 30.0, self.frame_size)
                    self.is_recording = True
                    # Also reset accumulated yaw when starting recording
                    self.accumulated_yaw = 0.0
                    self.last_yaw = None
                    self.get_logger().info(f"Started recording: {self.video_filename} (Rotation RESET)")
                else:
                    # Stop recording
                    if self.video_writer is not None:
                        self.video_writer.release()
                        self.video_writer = None
                    self.is_recording = False
                    self.get_logger().info(f"Stopped recording: {self.video_filename}")
            
            # 'T' or 't' to reset accumulated rotation only
            if key == ord('t') or key == ord('T'):
                self.accumulated_yaw = 0.0
                self.last_yaw = None
                self.get_logger().info("Accumulated rotation RESET by user.")
            
            # Write frame to video if recording
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(cv_image)

    def detect_rotation_matrix_to_quaternion(self, m):
        # Implementation of rotation matrix to quaternion
        # m is 3x3 numpy array
        tr = m[0,0] + m[1,1] + m[2,2]
        if tr > 0:
            S = np.sqrt(tr+1.0) * 2
            qw = 0.25 * S
            qx = (m[2,1] - m[1,2]) / S
            qy = (m[0,2] - m[2,0]) / S
            qz = (m[1,0] - m[0,1]) / S
        elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            qw = (m[2,1] - m[1,2]) / S
            qx = 0.25 * S
            qy = (m[0,1] + m[1,0]) / S
            qz = (m[0,2] + m[2,0]) / S
        elif (m[1,1] > m[2,2]):
            S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            qw = (m[0,2] - m[2,0]) / S
            qx = (m[0,1] + m[1,0]) / S
            qy = 0.25 * S
            qz = (m[1,2] + m[2,1]) / S
        else:
            S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            qw = (m[1,0] - m[0,1]) / S
            qx = (m[0,2] + m[2,0]) / S
            qy = (m[1,2] + m[2,1]) / S
            qz = 0.25 * S
        return [qx, qy, qz, qw]

    def destroy_node(self):
        # Stop video recording if active
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f"Recording saved: {self.video_filename}")
        if hasattr(self, 'csv_file') and self.csv_file:
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
