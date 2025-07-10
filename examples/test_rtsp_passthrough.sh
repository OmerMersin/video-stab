#!/bin/bash

# Test RTSP passthrough with interpipes
echo "Testing RTSP passthrough with interpipes..."

# Check if we have the required elements
if ! gst-inspect-1.0 interpipesrc >/dev/null 2>&1; then
    echo "Error: interpipesrc element not found. Please install gst-interpipe."
    exit 1
fi

if ! gst-inspect-1.0 rtspclientsink >/dev/null 2>&1; then
    echo "Error: rtspclientsink element not found. Please install gst-rtsp or gst-plugins-bad."
    exit 1
fi

# Kill any existing gstd instances
pkill -f gstd

# Start gstd
echo "Starting gstd..."
gstd --daemon

sleep 2

echo "Creating RTSP passthrough test..."

# Create passthrough pipeline (simulates RTSP source)
gst-client pipeline_create rtsp_source "videotestsrc pattern=0 ! video/x-raw,width=640,height=480,framerate=30/1 ! x264enc tune=zerolatency ! h264parse ! interpipesink name=passthrough_out sync=false"

# Create output pipeline (sends to RTSP sink) 
gst-client pipeline_create rtsp_output "interpipesrc name=src listen-to=passthrough_out is-live=true ! queue ! rtph264pay config-interval=1 ! udpsink host=127.0.0.1 port=5000"

# Start both pipelines
echo "Starting pipelines..."
gst-client pipeline_play rtsp_source
gst-client pipeline_play rtsp_output

echo "Test running for 10 seconds..."
echo "You can test with: gst-launch-1.0 udpsrc port=5000 ! 'application/x-rtp, encoding-name=H264' ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink"

sleep 10

# Stop and cleanup
echo "Stopping pipelines..."
gst-client pipeline_stop rtsp_output
gst-client pipeline_stop rtsp_source
gst-client pipeline_delete rtsp_output
gst-client pipeline_delete rtsp_source
gst-client quit

echo "Test complete."
