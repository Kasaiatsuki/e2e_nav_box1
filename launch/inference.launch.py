import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('e2e_nav_box1')
    config_file = os.path.join(pkg_share, 'config', 'inference_node.yaml')

    inference_node = Node(
        package='e2e_nav_box1',
        executable='inference_node',
        name='inference_node',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        inference_node
    ])
