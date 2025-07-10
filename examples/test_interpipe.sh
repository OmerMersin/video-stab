#!/bin/bash

# Test script for gstd interpipe setup
echo "Testing gstd interpipe setup..."

# Kill any existing gstd instances
pkill -f gstd

# Start gstd
echo "Starting gstd..."
gstd --daemon

sleep 2

# Test basic interpipe setup
echo "Creating test pipelines..."

# Create a test source pipeline
gst-client pipeline_create test_source "videotestsrc pattern=0 ! video/x-raw,width=640,height=480,framerate=30/1 ! interpipesink name=test_out sync=false"

# Create a test sink pipeline  
gst-client pipeline_create test_sink "interpipesrc listen-to=test_out is-live=true ! videoconvert ! autovideosink"

# Start both pipelines
echo "Starting pipelines..."
gst-client pipeline_play test_source
gst-client pipeline_play test_sink

echo "Test running for 10 seconds..."
sleep 10

# Stop and cleanup
echo "Stopping pipelines..."
gst-client pipeline_stop test_sink
gst-client pipeline_stop test_source
gst-client pipeline_delete test_sink
gst-client pipeline_delete test_source
gst-client quit

echo "Test complete."
