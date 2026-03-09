from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'e2e_nav_box1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # config や weights フォルダもインストール先へコピーするための設定
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Atsuki Kasai',
    maintainer_email='kasaiatuski@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = e2e_nav_box1.inference_node:main',
            'train = e2e_nav_box1.train:main',
            'create_data = e2e_nav_box1.create_data:main',
        ],
    },
)
