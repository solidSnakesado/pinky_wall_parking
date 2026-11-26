from glob import glob

from setuptools import find_packages, setup

package_name = 'pinky_wall_parking_maneuver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sado',
    maintainer_email='52153542+solidSnakesado@users.noreply.github.com',
    description='Car-like parking maneuvers (Reeds-Shepp, T parking)',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            't_parking_node = pinky_wall_parking_maneuver.t_parking_node:main',
            'slot_parallel_node = pinky_wall_parking_maneuver.slot_parallel_node:main',
        ],
    },
)
