#!/bin/bash

# Simple RTSP viewer to test the output stream

echo "=========================================="
echo "RTSP Stream Viewer Test"
echo "=========================================="

RTSP_URL="rtsp://127.0.0.1:8554/forwarded"

echo "Testing RTSP stream: $RTSP_URL"
echo ""

# Method 1: Using GStreamer playbin
echo "Method 1: GStreamer playbin"
echo "gst-launch-1.0 playbin uri=$RTSP_URL"
echo ""

# Method 2: Using GStreamer rtspsrc
echo "Method 2: GStreamer rtspsrc pipeline"
echo "gst-launch-1.0 rtspsrc location=$RTSP_URL ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink"
echo ""

# Method 3: Using VLC (if available)
if command -v vlc &> /dev/null; then
    echo "Method 3: VLC Player"
    echo "vlc $RTSP_URL"
    echo ""
fi

# Method 4: Using ffplay (if available)
if command -v ffplay &> /dev/null; then
    echo "Method 4: FFplay"
    echo "ffplay $RTSP_URL"
    echo ""
fi

echo "Choose a method to test the RTSP stream:"
echo "1) GStreamer playbin"
echo "2) GStreamer rtspsrc pipeline" 
echo "3) VLC (if installed)"
echo "4) FFplay (if installed)"
echo "5) Just show commands and exit"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Starting GStreamer playbin..."
        gst-launch-1.0 playbin uri=$RTSP_URL
        ;;
    2)
        echo "Starting GStreamer rtspsrc pipeline..."
        gst-launch-1.0 rtspsrc location=$RTSP_URL ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
        ;;
    3)
        if command -v vlc &> /dev/null; then
            echo "Starting VLC..."
            vlc $RTSP_URL
        else
            echo "VLC not found"
        fi
        ;;
    4)
        if command -v ffplay &> /dev/null; then
            echo "Starting FFplay..."
            ffplay $RTSP_URL
        else
            echo "FFplay not found"
        fi
        ;;
    5)
        echo "Commands shown above. Use any of them to test the RTSP stream."
        ;;
    *)
        echo "Invalid choice"
        ;;
esac
