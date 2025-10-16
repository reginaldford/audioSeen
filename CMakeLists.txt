cmake_minimum_required(VERSION 3.10)
project(alsa_eigen_fft LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(FFTW3 REQUIRED COMPONENTS double)

# Eigen3 (header-only)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find ALSA library using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(ALSA REQUIRED alsa)

# Define the executable
add_executable(alsa_eigen_fft
    main.cpp
)

# Include directories
target_include_directories(alsa_eigen_fft PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${ALSA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(alsa_eigen_fft PRIVATE
    ${ALSA_LIBRARIES}   # Link to ALSA library
    FFTW3::fftw3        # Link to FFTW3
)

# Compiler flags (from pkg-config)
target_compile_options(alsa_eigen_fft PRIVATE ${ALSA_CFLAGS_OTHER})

