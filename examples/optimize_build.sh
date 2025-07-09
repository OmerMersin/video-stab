#!/bin/bash

# Optimized build script for Jetson Orin Nano
# This script builds the project with optimizations for ARM processors

echo "Building libtracker with ARM optimizations..."

# Set compiler flags for ARM64 optimization (Jetson Orin Nano)
export CXXFLAGS="-O3 -march=armv8.2-a -mtune=cortex-a78 -ftree-vectorize -ffast-math -funroll-loops"
export CFLAGS="-O3 -march=armv8.2-a -mtune=cortex-a78 -ftree-vectorize -ffast-math -funroll-loops"

# Build with optimizations
cd /home/orin/Developer/libtracker
mkdir -p build
cd build

# Configure with optimizations
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
      -DCMAKE_C_FLAGS="${CFLAGS}" \
      -DWITH_CUDA=ON \
      -Wno-dev \
      ..

# Build
make -j$(nproc)

echo "Build complete!"
echo "Performance optimizations applied:"
echo "- ARM v8.2-a architecture targeting"
echo "- Cortex-A78 tuning (Orin Nano CPU)"
echo "- NEON SIMD instructions (native ARM64)"
echo "- Tree vectorization enabled"
echo "- Fast math optimizations enabled"
echo "- Loop unrolling enabled"
echo "- Release build with O3 optimization"
