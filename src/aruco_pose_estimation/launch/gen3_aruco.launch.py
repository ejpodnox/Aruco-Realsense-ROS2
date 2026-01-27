# ROS2 imports
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # Declare arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.10',
        description='IP address of the robot'
    )

    use_fake_hardware_arg = DeclareLaunchArgument(
        'use_fake_hardware',
        default_value='true',
        description='Use fake hardware for simulation'
    )

    launch_camera_arg = DeclareLaunchArgument(
        'launch_camera',
        default_value='false',
        description='Launch RealSense camera'
    )

    marker_size_arg = DeclareLaunchArgument(
        'marker_size',
        default_value='0.05',
        description='Size of ArUco markers in meters'
    )

    # Launch Kinova Gen3 robot
    gen3_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('kortex_bringup'),
                'launch',
                'gen3.launch.py'
            ])
        ),
        launch_arguments={
            'robot_ip': LaunchConfiguration('robot_ip'),
            'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
        }.items()
    )

    # Launch ArUco pose estimation
    aruco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('aruco_pose_estimation'),
                'launch',
                'aruco_pose_estimation.launch.py'
            ])
        ),
        launch_arguments={
            'launch_camera': LaunchConfiguration('launch_camera'),
            'marker_size': LaunchConfiguration('marker_size'),
        }.items()
    )

    return LaunchDescription([
        # Arguments
        robot_ip_arg,
        use_fake_hardware_arg,
        launch_camera_arg,
        marker_size_arg,

        # Launch files
        gen3_launch,
        aruco_launch,
    ])
