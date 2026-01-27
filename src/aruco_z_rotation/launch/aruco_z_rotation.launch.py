from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    marker_id_arg = DeclareLaunchArgument('marker_id', default_value='0', description='ID of the ArUco marker to track')
    marker_size_arg = DeclareLaunchArgument('marker_size', default_value='0.05', description='Size of the marker in meters')
    
    return LaunchDescription([
        marker_id_arg,
        marker_size_arg,
        Node(
            package='aruco_z_rotation',
            executable='z_rotation_node',
            name='aruco_z_rotation_node',
            output='screen',
            parameters=[{
                'marker_id': LaunchConfiguration('marker_id'),
                'marker_size': LaunchConfiguration('marker_size'),
                'camera_topic': '/camera/color/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'dictionary_id': 'DICT_4X4_50'
            }]
        )
    ])
