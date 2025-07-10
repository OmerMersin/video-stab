#!/bin/bash

# Test script for gstd interpipe system

echo "=========================================="
echo "GStreamer Daemon Interpipe Test"
echo "=========================================="

# Check if required components are available
echo "Checking prerequisites..."

if ! command -v gstd &> /dev/null; then
    echo "❌ gstd not found"
    exit 1
else
    echo "✅ gstd found"
fi

if ! gst-inspect-1.0 interpipesrc &> /dev/null; then
    echo "❌ interpipe plugin not found"
    exit 1
else
    echo "✅ interpipe plugin found"
fi

if ! gst-inspect-1.0 rtspclientsink &> /dev/null; then
    echo "❌ rtspclientsink not found"
    exit 1
else
    echo "✅ rtspclientsink found"
fi

echo ""
echo "Building application..."
cd /home/orin/Developer/libtracker
./examples/build_gstd_interpipe.sh

if [ ! -f "examples/main-gstd-interpipe" ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build successful"
echo ""

# Start RTSP server in background to receive the stream
echo "Starting test RTSP server..."
echo "Note: You can also test by connecting an RTSP viewer to rtsp://127.0.0.1:8554/forwarded"

# Kill any existing gstd instances
pkill -f gstd >/dev/null 2>&1
sleep 1

echo ""
echo "Starting main application..."
echo "Press Ctrl+C to stop"
echo ""

# Run the application
./examples/main-gstd-interpipe examples/config_gstd_interpipe.yaml rtsp://127.0.0.1:8554/forwarded

echo ""
echo "Test completed."
