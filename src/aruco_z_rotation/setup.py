import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'aruco_z_rotation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/aruco_z_rotation.launch.py', 'launch/aruco_with_realsense.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/aruco_vis.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='billy-linux-think',
    maintainer_email='billy-linux-think@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'z_rotation_node = aruco_z_rotation.z_rotation_node:main',
        ],
    },
)
