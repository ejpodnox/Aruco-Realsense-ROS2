import cv2
import numpy as np
import pytest

from aruco_pose_estimation.board_fusion import (
    estimate_fused_board_pose,
    get_local_marker_corners,
    load_board_configuration,
)


def test_load_board_configuration_generates_expected_marker_corners(tmp_path):
    config_path = tmp_path / "board.yaml"
    config_path.write_text(
        """
aruco_board:
  frame_name: board
  markers:
    - id: 0
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
""".strip(),
        encoding="utf-8",
    )

    board_config = load_board_configuration(str(config_path), marker_size=0.1)

    np.testing.assert_allclose(
        board_config.markers[0].object_points,
        get_local_marker_corners(0.1),
        atol=1e-7,
    )


def test_load_board_configuration_rejects_duplicate_marker_ids(tmp_path):
    config_path = tmp_path / "duplicate_board.yaml"
    config_path.write_text(
        """
aruco_board:
  markers:
    - id: 0
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
    - id: 0
      xyz: [0.1, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate board marker id"):
        load_board_configuration(str(config_path), marker_size=0.1)


def test_estimate_fused_board_pose_recovers_known_pose(tmp_path):
    config_path = tmp_path / "board_pose.yaml"
    config_path.write_text(
        """
aruco_board:
  frame_name: board
  markers:
    - id: 0
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
    - id: 1
      xyz: [0.08, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
""".strip(),
        encoding="utf-8",
    )

    board_config = load_board_configuration(str(config_path), marker_size=0.05)
    camera_matrix = np.array(
        [[610.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    distortion = np.zeros((5,), dtype=np.float32)
    true_rvec = np.array([[0.12], [-0.05], [0.08]], dtype=np.float32)
    true_tvec = np.array([[0.02], [0.01], [0.75]], dtype=np.float32)

    corners = []
    marker_ids = []
    for marker_id in sorted(board_config.markers):
        projected_points, _ = cv2.projectPoints(
            board_config.markers[marker_id].object_points,
            true_rvec,
            true_tvec,
            camera_matrix,
            distortion,
        )
        corners.append(projected_points.reshape(1, 4, 2).astype(np.float32))
        marker_ids.append([marker_id])

    board_status = estimate_fused_board_pose(
        corners=corners,
        marker_ids=np.array(marker_ids, dtype=np.int32),
        board_config=board_config,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        min_markers=2,
        refine=True,
    )

    assert board_status.pose_estimate is not None
    np.testing.assert_allclose(board_status.pose_estimate.translation, true_tvec.reshape(3), atol=1e-4)

    estimated_rotation, _ = cv2.Rodrigues(board_status.pose_estimate.rotation_vector)
    true_rotation, _ = cv2.Rodrigues(true_rvec)
    np.testing.assert_allclose(estimated_rotation, true_rotation, atol=1e-4)
    assert board_status.pose_estimate.reprojection_error is not None
    assert board_status.pose_estimate.reprojection_error < 1e-4


def test_estimate_fused_board_pose_requires_minimum_visible_markers(tmp_path):
    config_path = tmp_path / "single_visible_board.yaml"
    config_path.write_text(
        """
aruco_board:
  frame_name: board
  markers:
    - id: 0
      xyz: [0.0, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
    - id: 1
      xyz: [0.08, 0.0, 0.0]
      rpy: [0.0, 0.0, 0.0]
""".strip(),
        encoding="utf-8",
    )

    board_config = load_board_configuration(str(config_path), marker_size=0.05)
    camera_matrix = np.array(
        [[610.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    distortion = np.zeros((5,), dtype=np.float32)
    true_rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    true_tvec = np.array([[0.0], [0.0], [0.7]], dtype=np.float32)

    projected_points, _ = cv2.projectPoints(
        board_config.markers[0].object_points,
        true_rvec,
        true_tvec,
        camera_matrix,
        distortion,
    )

    board_status = estimate_fused_board_pose(
        corners=[projected_points.reshape(1, 4, 2).astype(np.float32)],
        marker_ids=np.array([[0]], dtype=np.int32),
        board_config=board_config,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        min_markers=2,
        refine=True,
    )

    assert board_status.visible_marker_ids == [0]
    assert board_status.pose_estimate is None
