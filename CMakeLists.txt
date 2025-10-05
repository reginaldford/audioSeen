cmake_minimum_required(VERSION 3.10)
project(pa_eigen_fft LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# PulseAudio
pkg_check_modules(PULSEAUDIO REQUIRED libpulse libpulse-simple)

# FFTW3
find_package(FFTW3 REQUIRED COMPONENTS double)

# Eigen3 (header-only)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Define the executable
add_executable(pa_eigen_fft
    main.cpp
)

# Include directories
target_include_directories(pa_eigen_fft PRIVATE
    ${PULSEAUDIO_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# Libraries to link
target_link_libraries(pa_eigen_fft PRIVATE
    ${PULSEAUDIO_LIBRARIES}
    FFTW3::fftw3
)

# Compiler flags (from pkg-config)
target_compile_options(pa_eigen_fft PRIVATE ${PULSEAUDIO_CFLAGS_OTHER})

