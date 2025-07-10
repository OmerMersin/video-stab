#!/bin/bash

# Build script for gstd-based interpipe streaming system

set -e

echo "Building gstd-based interpipe streaming system..."

# Check if gstd is installed
if ! command -v gstd &> /dev/null; then
    echo "Warning: gstd not found. Please install gstreamer1.0-daemon"
    echo "sudo apt-get install gstreamer1.0-daemon"
fi

# Check if interpipe plugin is available
if ! gst-inspect-1.0 interpipesrc &> /dev/null; then
    echo "Warning: interpipe plugin not found. Please install gst-interpipe"
    echo "https://github.com/RidgeRun/gst-interpipe"
fi

# Build directory
BUILD_DIR="build"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure cmake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17

# Build the library
make -j$(nproc)

cd ..

# Build the application
echo "Building main-gstd-interpipe application..."

g++ -std=c++17 -O2 \
    -I./include \
    -I/usr/local/include \
    -I/opt/nvidia/deepstream/deepstream-7.1/sources/includes \
    examples/main-gstd-interpipe.cpp \
    -L./build \
    -L/usr/local/lib \
    -L/opt/nvidia/deepstream/deepstream-7.1/lib \
    -lvideo-stab \
    $(pkg-config --cflags --libs opencv4) \
    $(pkg-config --cflags --libs gstreamer-1.0) \
    $(pkg-config --cflags --libs gstreamer-app-1.0) \
    $(pkg-config --cflags --libs gstreamer-rtsp-server-1.0) \
    $(pkg-config --cflags --libs glib-2.0) \
    $(pkg-config --cflags --libs gobject-2.0) \
    -lnvdsgst_meta \
    -lnvds_meta \
    -lnvds_inferutils \
    -lnvds_infercustomparser \
    -lnvbufsurface \
    -lnvbufsurftransform \
    -lpthread \
    -o examples/main-gstd-interpipe

if [ -f "examples/main-gstd-interpipe" ]; then
    echo "Built executable: examples/main-gstd-interpipe"
else
    echo "Error: Build failed"
    exit 1
fi

echo "Build complete!"
echo ""
echo "Usage:"
echo "  ./examples/main-gstd-interpipe config.yaml rtsp://localhost:8554/stream"
echo ""
echo "Make sure to:"
echo "1. Install gstreamer1.0-daemon: sudo apt-get install gstreamer1.0-daemon"
echo "2. Install gst-interpipe plugin"
echo "3. Configure your RTSP server to accept streams at the specified URL"
