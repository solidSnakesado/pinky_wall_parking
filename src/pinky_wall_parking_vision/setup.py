from glob import glob

from setuptools import find_packages, setup

package_name = 'pinky_wall_parking_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sado',
    maintainer_email='52153542+solidSnakesado@users.noreply.github.com',
    description='Camera slot features for front parking (IBVS)',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'slot_vision_node = pinky_wall_parking_vision.slot_vision_node:main',
            'vision_bag_eval = pinky_wall_parking_vision.vision_bag_eval:main',
        ],
    },
)
