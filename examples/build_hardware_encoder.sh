#!/bin/bash

# Build script for hardware encoder optimized version
echo "Building hardware encoder optimized version..."

# Set environment variables
export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
export JETSON_MULTIMEDIA_API_PATH=/usr/src/jetson_multimedia_api

# Create build directory
mkdir -p build_hardware_encoder
cd build_hardware_encoder

# Configure with cmake
cmake -DCMAKE_BUILD_TYPE=Release -f ../CMakeLists_hardware_encoder.txt ..

# Build
make -j$(nproc)

echo "Build completed!"
echo "Executable: ./hardware_encoder_example"
echo ""
echo "Usage: ./hardware_encoder_example config.yaml"
