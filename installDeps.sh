#!/usr/bin/env bash
# Works for Fedora 38
set -e

echo "Updating system..."
sudo dnf -y update

echo "Installing development tools and required libraries..."
sudo dnf -y install \
    gcc \
    gcc-c++ \
    make \
    cmake \
    pkg-config \
    pulseaudio-libs-devel \
    fftw-devel \
    eigen3-devel

echo "All dependencies installed successfully!"

