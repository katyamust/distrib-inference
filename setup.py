
from setuptools import setup, find_packages

__version__ = "1.0"#
#with open('VERSION') as version_file:
#    __version__ = version_file.read().strip()

setup(name='distrib_resnet',
      version=__version__,
      description='distrib inference',
      long_description="",
      long_description_content_type='text/markdown',
      license='MIT',
      packages=find_packages()
      )
