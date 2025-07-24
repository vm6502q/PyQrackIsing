cmake_minimum_required(VERSION 3.15...3.29)
project(PyQrackIsing LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tfim_sampler PyQrackIsing/_pyqrack_ising.cpp)

if (DEFINED ENV{BOOST_ROOT})
  target_include_directories(_find_a_factor PUBLIC $ENV{BOOST_ROOT}/include/boost)
endif (DEFINED ENV{BOOST_ROOT})

if (MSVC)
  target_compile_options(tfim_sampler PUBLIC /O2 /std:c++17)
else (MSVC)
  target_compile_options(tfim_sampler PUBLIC -O3 -std=c++17)
endif (MSVC)

install(TARGETS tfim_sampler DESTINATION .)
