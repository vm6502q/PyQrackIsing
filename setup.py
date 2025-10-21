from setuptools import setup
import os

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(
    name='pyqrackising',
    version='8.14.8',
    author='Dan Strano',
    author_email='stranoj@gmail.com',
    description='Fast MAXCUT, TSP, and sampling heuristics from near-ideal transverse field Ising model (TFIM)',
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/vm6502q/PyQrackIsing",
    license="LGPL-3.0-or-later",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    packages=['pyqrackising'],
    zip_safe=False,
)
