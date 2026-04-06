#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class BoardMarkerDefinition:
    marker_id: int
    xyz: np.ndarray
    rpy: np.ndarray
    object_points: np.ndarray


@dataclass(frozen=True)
class BoardConfiguration:
    frame_name: str
    markers: dict[int, BoardMarkerDefinition]


@dataclass(frozen=True)
class BoardPoseEstimate:
    translation: np.ndarray
    rotation_vector: np.ndarray
    quaternion: np.ndarray
    reprojection_error: Optional[float]


@dataclass(frozen=True)
class BoardFusionStatus:
    frame_name: str
    visible_marker_ids: list[int]
    min_required_markers: int
    pose_estimate: Optional[BoardPoseEstimate]


def load_board_configuration(config_path: str, marker_size: float) -> BoardConfiguration:
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"Board config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    board_config = config.get("aruco_board")
    if not isinstance(board_config, dict):
        raise ValueError("Board config must contain an 'aruco_board' mapping.")

    frame_name = board_config.get("frame_name", "board")
    if not isinstance(frame_name, str) or not frame_name.strip():
        raise ValueError("Board config 'frame_name' must be a non-empty string.")

    markers_config = board_config.get("markers")
    if not isinstance(markers_config, list) or not markers_config:
        raise ValueError("Board config must define a non-empty 'markers' list.")

    markers: dict[int, BoardMarkerDefinition] = {}
    for index, marker_config in enumerate(markers_config):
        if not isinstance(marker_config, dict):
            raise ValueError(f"Board marker entry at index {index} must be a mapping.")

        marker_id = marker_config.get("id")
        if not isinstance(marker_id, int):
            raise ValueError(f"Board marker entry at index {index} must define an integer 'id'.")
        if marker_id in markers:
            raise ValueError(f"Duplicate board marker id found: {marker_id}")

        xyz = _parse_vector(marker_config.get("xyz"), "xyz", marker_id)
        rpy = _parse_vector(marker_config.get("rpy"), "rpy", marker_id)
        object_points = generate_marker_object_points(marker_size, xyz, rpy)

        markers[marker_id] = BoardMarkerDefinition(
            marker_id=marker_id,
            xyz=xyz,
            rpy=rpy,
            object_points=object_points,
        )

    return BoardConfiguration(frame_name=frame_name.strip(), markers=markers)


def generate_marker_object_points(marker_size: float, xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    marker_points = get_local_marker_corners(marker_size)
    rotation_matrix = rotation_matrix_from_rpy(rpy[0], rpy[1], rpy[2])
    transformed_points = (rotation_matrix @ marker_points.T).T + xyz
    return transformed_points.astype(np.float32)


def get_local_marker_corners(marker_size: float) -> np.ndarray:
    half_size = marker_size / 2.0
    return np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float32,
    )


def collect_board_observations(
    corners: list[np.ndarray],
    marker_ids: Optional[np.ndarray],
    board_config: BoardConfiguration,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if marker_ids is None or len(corners) == 0:
        return _empty_board_observations()

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    visible_marker_ids: list[int] = []

    for marker_corners, marker_id in zip(corners, np.asarray(marker_ids).reshape(-1)):
        marker_id = int(marker_id)
        marker_definition = board_config.markers.get(marker_id)
        if marker_definition is None:
            continue

        object_points.append(marker_definition.object_points)
        image_points.append(np.asarray(marker_corners, dtype=np.float32).reshape(4, 2))
        visible_marker_ids.append(marker_id)

    if not object_points:
        return _empty_board_observations()

    return (
        np.concatenate(object_points, axis=0).astype(np.float32),
        np.concatenate(image_points, axis=0).astype(np.float32),
        visible_marker_ids,
    )


def estimate_fused_board_pose(
    corners: list[np.ndarray],
    marker_ids: Optional[np.ndarray],
    board_config: BoardConfiguration,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    min_markers: int = 2,
    refine: bool = True,
) -> BoardFusionStatus:
    object_points, image_points, visible_marker_ids = collect_board_observations(
        corners,
        marker_ids,
        board_config,
    )

    if len(visible_marker_ids) < min_markers or len(object_points) < 4:
        return BoardFusionStatus(
            frame_name=board_config.frame_name,
            visible_marker_ids=visible_marker_ids,
            min_required_markers=min_markers,
            pose_estimate=None,
        )

    success, rvec, tvec, _ = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return BoardFusionStatus(
            frame_name=board_config.frame_name,
            visible_marker_ids=visible_marker_ids,
            min_required_markers=min_markers,
            pose_estimate=None,
        )

    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    if refine and hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
            rvec=rvec,
            tvec=tvec,
        )

    quaternion = rotation_vector_to_quaternion(rvec)
    reprojection_error = compute_reprojection_error(
        object_points,
        image_points,
        camera_matrix,
        distortion_coefficients,
        rvec,
        tvec,
    )

    return BoardFusionStatus(
        frame_name=board_config.frame_name,
        visible_marker_ids=visible_marker_ids,
        min_required_markers=min_markers,
        pose_estimate=BoardPoseEstimate(
            translation=tvec.reshape(3),
            rotation_vector=rvec.reshape(3, 1),
            quaternion=quaternion,
            reprojection_error=reprojection_error,
        ),
    )


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector.reshape(3, 1))
    quaternion = quaternion_from_rotation_matrix(rotation_matrix)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        raise ValueError("Quaternion normalization failed for board pose estimate.")
    return quaternion / norm


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def quaternion_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    trace = np.trace(matrix)

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    return np.array([qx, qy, qz, qw], dtype=np.float32)


def compute_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    rotation_vector: np.ndarray,
    translation: np.ndarray,
) -> float:
    projected_points, _ = cv2.projectPoints(
        objectPoints=object_points,
        rvec=rotation_vector,
        tvec=translation,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
    )
    projected_points = projected_points.reshape(-1, 2)
    residuals = projected_points - image_points.reshape(-1, 2)
    squared_distances = np.sum(residuals ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_distances)))


def _parse_vector(raw_value: object, name: str, marker_id: int) -> np.ndarray:
    if not isinstance(raw_value, list) or len(raw_value) != 3:
        raise ValueError(
            f"Board marker {marker_id} must define '{name}' as a list of three numeric values."
        )

    try:
        vector = np.array([float(value) for value in raw_value], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Board marker {marker_id} contains non-numeric values in '{name}'."
        ) from exc

    return vector


def _empty_board_observations() -> tuple[np.ndarray, np.ndarray, list[int]]:
    return (
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        [],
    )
