from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            import cmake
        except ImportError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Release'
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
        ]

        cmake_args = ['-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}']
        toolchain = os.environ.get("BOOST_TOOLCHAIN_FILE")
        if toolchain:
            cmake_args += [f'-DCMAKE_TOOLCHAIN_FILE={toolchain}']
        root = os.environ.get("BOOST_ROOT")
        if root:
            cmake_args += ['-DBoost_NO_SYSTEM_PATHS=TRUE', f'-DBOOST_ROOT={root}']
        include = os.environ.get("BOOST_INCLUDEDIR")
        if include:
            cmake_args += [f'-DBoost_INCLUDE_DIR={include}']

        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        self.spawn(['cmake', '-B', build_temp, '-S', ext.sourcedir] + cmake_args)
        self.spawn(['cmake', '--build', build_temp, '--config', cfg])

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

ext_modules = [CMakeExtension('tfim_sampler')]

setup(
    name='PyQrackIsing',
    version='1.10.7',
    author='Dan Strano',
    author_email='stranoj@gmail.com',
    description='Near-ideal closed-form solutions for transverse field Ising model (TFIM)',
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
    install_requires=["pybind11"],
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    packages=['PyQrackIsing'],
    zip_safe=False,
)
