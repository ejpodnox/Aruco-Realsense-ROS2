#!/usr/bin/env python3

# Code taken and readapted from:
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/tree/main

# Python imports
import numpy as np
import cv2
from typing import Any, Optional

# ROS2 imports
from rclpy.impl import rcutils_logger

# ROS2 message imports
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from aruco_interfaces.msg import ArucoMarkers

# utils import python code
from aruco_pose_estimation.board_fusion import (
    BoardConfiguration,
    BoardFusionStatus,
    estimate_fused_board_pose,
    rotation_vector_to_quaternion,
)
from aruco_pose_estimation.utils import aruco_display, draw_text_block

# Reuse a single logger to avoid recreating it every frame
logger = rcutils_logger.RcutilsLogger(name="aruco_node")


def pose_estimation(rgb_frame: np.ndarray, depth_frame: Optional[np.ndarray], aruco_detector: Optional[Any], marker_size: float,
                    matrix_coefficients: np.ndarray, distortion_coefficients: np.ndarray,
                    pose_array: PoseArray, markers: ArucoMarkers,
                    aruco_dictionary: Optional[Any] = None, aruco_parameters: Optional[Any] = None,
                    board_config: Optional[BoardConfiguration] = None,
                    board_min_markers: int = 2,
                    board_pose_refine: bool = True) -> tuple[np.ndarray, PoseArray, ArucoMarkers, Optional[BoardFusionStatus]]:
    '''
    rgb_frame - Frame from the RGB camera stream
    depth_frame - Depth frame from the depth camera stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    pose_array - PoseArray message to be published
    markers - ArucoMarkers message to be published

    return:-
    frame - The frame with the axis drawn on it
    pose_array - PoseArray with computed poses of the markers
    markers - ArucoMarkers message containing markers id number and pose
    '''

    # old code version
    # parameters = cv2.aruco.DetectorParameters_create()
    # corners, marker_ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict_type, parameters=parameters)

    # new code version
    detection_frame = rgb_frame
    if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 3:
        detection_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    # Support both APIs:
    # - OpenCV >= 4.7 style: cv2.aruco.ArucoDetector
    # - OpenCV legacy style: cv2.aruco.detectMarkers
    if aruco_detector is not None and hasattr(aruco_detector, "detectMarkers"):
        corners, marker_ids, rejected = aruco_detector.detectMarkers(image=detection_frame)
    else:
        if aruco_dictionary is None or aruco_parameters is None:
            raise ValueError("Legacy OpenCV ArUco API requires dictionary and detector parameters.")
        corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            detection_frame,
            aruco_dictionary,
            parameters=aruco_parameters,
        )

    frame_processed = rgb_frame

    board_status = None

    # If markers are detected
    if len(corners) > 0:

        logger.debug("Detected {} markers.".format(len(corners)))

        # draw the detected markers once outside the loop to avoid redundant work
        frame_processed = aruco_display(corners=corners, ids=marker_ids,
                                        image=frame_processed)

        for i, marker_id in enumerate(marker_ids):
            # Estimate pose of each marker and return the values rvec and tvec

            # using deprecated function
            # rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners=corners[i],
            #                                                               markerLength=marker_size,
            #                                                               cameraMatrix=matrix_coefficients,
            #                                                               distCoeffs=distortion_coefficients)
            # tvec = tvec[0]

            # alternative code version using solvePnP
            tvec, rvec, quat = my_estimatePoseSingleMarkers(corners=corners[i], marker_size=marker_size,
                                                                    camera_matrix=matrix_coefficients,
                                                                    distortion=distortion_coefficients)

            # draw frame axes
            frame_processed = cv2.drawFrameAxes(image=frame_processed, cameraMatrix=matrix_coefficients,
                                                distCoeffs=distortion_coefficients, rvec=rvec, tvec=tvec,
                                                length=0.05, thickness=3)

            centroid = None
            if (depth_frame is not None):
                # get the centroid of the pointcloud
                centroid = depth_to_pointcloud_centroid(depth_image=depth_frame,
                                                        intrinsic_matrix=matrix_coefficients,
                                                        corners=corners[i])

            # compute pose from the rvec and tvec arrays
            if depth_frame is not None and centroid is not None:
                # use computed centroid from depthcloud as estimated pose
                pose = Pose()
                pose.position.x = float(centroid[0])
                pose.position.y = float(centroid[1])
                pose.position.z = float(centroid[2])
            else:
                # use tvec from aruco estimator as estimated pose
                pose = Pose()
                pose.position.x = float(tvec[0])
                pose.position.y = float(tvec[1])
                pose.position.z = float(tvec[2])

            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            # add the pose and marker id to the pose_array and markers messages
            pose_array.poses.append(pose)
            markers.poses.append(pose)
            markers.marker_ids.append(int(marker_id[0]))

    if board_config is not None:
        board_status = estimate_fused_board_pose(
            corners=corners,
            marker_ids=marker_ids,
            board_config=board_config,
            camera_matrix=matrix_coefficients,
            distortion_coefficients=distortion_coefficients,
            min_markers=board_min_markers,
            refine=board_pose_refine,
        )

        frame_processed = draw_board_fusion_status(
            image=frame_processed,
            board_status=board_status,
            marker_size=marker_size,
            camera_matrix=matrix_coefficients,
            distortion_coefficients=distortion_coefficients,
        )

    return frame_processed, pose_array, markers, board_status


def my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion) -> tuple[np.array, np.array, np.array]:
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)

    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers in meters
    mtx - is the camera intrinsic matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2.0, marker_size / 2.0, 0],
                              [marker_size / 2.0, marker_size / 2.0, 0],
                              [marker_size / 2.0, -marker_size / 2.0, 0],
                              [-marker_size / 2.0, -marker_size / 2.0, 0]], dtype=np.float32)

    # solvePnP returns the rotation and translation vectors
    retval, rvec, tvec = cv2.solvePnP(objectPoints=marker_points, imagePoints=corners,
                                        cameraMatrix=camera_matrix, distCoeffs=distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)
       
    quaternion = rotation_vector_to_quaternion(rvec)

    return tvec, rvec, quaternion


def depth_to_pointcloud_centroid(depth_image: np.array, intrinsic_matrix: np.array,
                                 corners: np.array) -> Optional[np.array]:
    """
    This function takes a depth image and the corners of a quadrilateral as input,
    and returns the centroid of the corresponding pointcloud.

    Args:
        depth_image: A 2D numpy array representing the depth image.
        corners: A list of 4 tuples, each representing the (x, y) coordinates of a corner.

    Returns:
        A tuple (x, y, z) representing the centroid of the segmented pointcloud.
        Returns None if no valid depth pixels are found inside the marker.
    """

    # Get image parameters
    height, width = depth_image.shape

    # corners has shape (1, 4, 2)
    corners_indices = np.round(corners[0]).astype(np.int32)

    if (
        np.any(corners_indices[:, 0] < 0)
        or np.any(corners_indices[:, 0] >= width)
        or np.any(corners_indices[:, 1] < 0)
        or np.any(corners_indices[:, 1] >= height)
    ):
        raise ValueError("One or more corners are outside the image bounds.")

    # bounding box of the polygon
    x_min = int(np.min(corners_indices[:, 0]))
    x_max = int(np.max(corners_indices[:, 0]))
    y_min = int(np.min(corners_indices[:, 1]))
    y_max = int(np.max(corners_indices[:, 1]))

    # Build a binary mask for the polygon inside its bounding box
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted_corners = corners_indices.copy()
    shifted_corners[:, 0] -= x_min
    shifted_corners[:, 1] -= y_min
    cv2.fillConvexPoly(mask, shifted_corners, 1)

    depth_roi = depth_image[y_min: y_max + 1, x_min: x_max + 1]

    valid_mask = (mask == 1) & (depth_roi > 0)
    if not np.any(valid_mask):
        return None

    ys, xs = np.nonzero(valid_mask)
    depths = depth_roi[valid_mask].astype(np.float32) * 0.001  # convert mm to meters

    # Map ROI coordinates back to image coordinates
    u = xs.astype(np.float32) + x_min
    v = ys.astype(np.float32) + y_min

    x = (u - intrinsic_matrix[0, 2]) * depths / intrinsic_matrix[0, 0]
    y = (v - intrinsic_matrix[1, 2]) * depths / intrinsic_matrix[1, 1]

    centroid = np.array([np.mean(x), np.mean(y), np.mean(depths)], dtype=np.float32)

    return centroid


def draw_board_fusion_status(
    image: np.ndarray,
    board_status: BoardFusionStatus,
    marker_size: float,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> np.ndarray:
    lines = [
        "Board fusion: {}/{} markers".format(
            len(board_status.visible_marker_ids),
            board_status.min_required_markers,
        ),
        "Board IDs: {}".format(board_status.visible_marker_ids if board_status.visible_marker_ids else "[]"),
    ]

    if board_status.pose_estimate is None:
        lines.append("Board pose unavailable")
        return draw_text_block(image, lines)

    image = cv2.drawFrameAxes(
        image=image,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
        rvec=board_status.pose_estimate.rotation_vector,
        tvec=board_status.pose_estimate.translation.reshape(3, 1),
        length=max(marker_size * 2.0, 0.05),
        thickness=4,
    )

    if board_status.pose_estimate.reprojection_error is not None:
        lines.append("Board RMS: {:.3f}px".format(board_status.pose_estimate.reprojection_error))

    return draw_text_block(image, lines)
