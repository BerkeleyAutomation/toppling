"""
Setup of Dex-Net python codebase
Author: Jeff Mahler
"""
from setuptools import setup

requirements = [
    # Our Repos
    'autolab-core',
    'meshrender',
    'visualization',

    # Contrib repos
    'trimesh[easy]',

    # External repos
    'ipython==5.5.0',
    'matplotlib',
    'numpy',
    'pybullet'
]

setup(name='toppling',
      version='0.1.0',
      description='Toppling project code',
      author='Chris Correa',
      author_email='chris.correa@berkeley.edu',
      package_dir = {'': 'src'},
      packages=['toppling'],
      install_requires=requirements,
      test_suite='test'
     )
