from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_aruco_z_rotation = FindPackageShare('aruco_z_rotation')
    pkg_realsense = FindPackageShare('realsense2_camera')

    # Arguments
    marker_id_arg = DeclareLaunchArgument('marker_id', default_value='0', description='ID of the ArUco marker to track')
    marker_size_arg = DeclareLaunchArgument('marker_size', default_value='0.03', description='Size of the marker in meters')
    
    # Common dictionaries: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, etc.
    dictionary_id_arg = DeclareLaunchArgument('dictionary_id', default_value='DICT_ORIGINAL', description='ArUco dictionary to use')
    
    show_window_arg = DeclareLaunchArgument('show_window', default_value='true', description='Show OpenCV window')
    csv_output_arg = DeclareLaunchArgument('csv_output', default_value='rotation_data.csv', description='CSV file to log data')

    # RealSense Camera Launch
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_realsense, 'launch', 'rs_launch.py'])
        ),
        # You might need to adjust these parameters for your specific RealSense model
        launch_arguments={
            'align_depth.enable': 'false',
            'enable_color': 'true',
            'enable_depth': 'false' # Depth not strictly needed for RGB detection
        }.items()
    )

    # ArUco Rotation Node
    aruco_node = Node(
        package='aruco_z_rotation',
        executable='z_rotation_node',
        name='aruco_z_rotation_node',
        output='screen',
        parameters=[{
            'marker_id': LaunchConfiguration('marker_id'),
            'marker_size': LaunchConfiguration('marker_size'),
            'camera_topic': '/camera/camera/color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'dictionary_id': LaunchConfiguration('dictionary_id'),
            'show_window': LaunchConfiguration('show_window'),
            'csv_output': LaunchConfiguration('csv_output')
        }]
    )

    # RViz2
    rviz_config_file = PathJoinSubstitution(
        [pkg_aruco_z_rotation, 'rviz', 'aruco_vis.rviz']
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )


    return LaunchDescription([
        marker_id_arg,
        marker_size_arg,
        dictionary_id_arg,
        show_window_arg,
        csv_output_arg,
        realsense_launch,
        aruco_node,
        rviz_node
    ])
