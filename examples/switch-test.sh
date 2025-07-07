#!/usr/bin/env bash
set -e

export CAM_URL="rtsp://192.168.144.119:554"
export LAT=150                          # jitter buffer for camera
export LOCAL_OUT="rtsp://127.0.0.1:8554/forwarded"   # ONE path, one word!

###############################################################################
# A) branch 1 – untouched colour H.265 straight from the camera
###############################################################################
gstd-client pipeline_create cam_raw "
  rtspsrc location=${CAM_URL} latency=${LAT} !
  rtph265depay ! h265parse ! 
  interpipesink name=raw
"
gstd-client pipeline_play cam_raw

###############################################################################
# B) branch 2 – desaturated and *re-encoded* back to H.265
###############################################################################
gstd-client pipeline_create cam_gray "
  rtspsrc location=${CAM_URL} latency=${LAT} !
  rtph265depay ! h265parse !
  avdec_h265 ! videoconvert !
  videobalance saturation=0 ! videoconvert !
  x265enc tune=zerolatency bitrate=4000 key-int-max=30 ! h265parse ! 
  interpipesink name=gray 
"
gstd-client pipeline_play cam_gray

###############################################################################
# C) sink – publishes ONE track, you flip which branch feeds it
###############################################################################
gstd-client pipeline_create cam_sink "
  interpipesrc name=selector listen-to=raw         \
               is-live=true do-timestamp=true      \
               allow-renegotiation=true !          \
  rtph265depay ! h265parse !            \
  rtspclientsink location=${LOCAL_OUT} protocols=tcp
"
gstd-client pipeline_play cam_sink
