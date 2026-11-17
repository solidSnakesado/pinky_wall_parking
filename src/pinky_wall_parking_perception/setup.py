from glob import glob
from setuptools import find_packages, setup

package_name = 'pinky_wall_parking_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='solidSnakesado',
    maintainer_email='52153542+solidSnakesado@users.noreply.github.com',
    description='Lidar wall slot detection for Pinky Pro',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "slot_detector = pinky_wall_parking_perception.slot_detector:main",
            "slot_eval = pinky_wall_parking_perception.slot_eval:main",
            "bag_eval = pinky_wall_parking_perception.bag_eval:main",
        ],
    },
)
