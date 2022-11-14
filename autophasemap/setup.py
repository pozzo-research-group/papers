from setuptools import setup,find_packages
import sys, os

setup(name="autophasemap",
      description="Polymer thermodynamic phase modelling",
      version='1.0',
      author='Kiran Vaddi',
      author_email='kiranvad@uw.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=[],
      extras_require = {},
      packages=find_packages(),
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
)