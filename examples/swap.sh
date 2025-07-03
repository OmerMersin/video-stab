#!/bin/bash
#
#  RTSP live-switch demo
#  Raw ⟷ Grayscale ⟷ Colour-with-Timestamp
#
#  Ctrl-C to stop.

CAM_URL="rtsp://192.168.144.119:554"
LOCAL_OUT="rtsp://127.0.0.1:8554/forwarded"   # gst-rtsp-server must be running
LAT=1000                                            # jitter buffer on the pull side

echo -e "\n ====== RTSP Switch Example (Raw ⇆ Gray ⇆ ClockOverlay) ====== \n"

STOP=0
trap "STOP=1" SIGINT

########################################################
# 1. three source pipelines – all ending in interpipesink
########################################################

echo -e "\n ✳️  pipe_1_src : RAW passthrough\n"
gstd-client pipeline_create pipe_1_src \
  "rtspsrc name=src location=${CAM_URL} latency=${LAT} ! \
   rtph265depay ! h265parse ! \
   interpipesink name=src_raw sync=false async=false drop=true"

echo -e "\n ✳️  pipe_2_src : GRAYSCALE (saturation 0)\n"
gstd-client pipeline_create pipe_2_src \
  "rtspsrc location=${CAM_URL} latency=${LAT} ! \
   rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! \
   videobalance saturation=0 ! videoconvert ! \
   x264enc threads=4 tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=15 intra-refresh=true ! \
   rtph264pay pt=97 config-interval=1 ! \
   interpipesink name=src_gray sync=false async=false drop=true"

echo -e "\n ✳️  pipe_3_src : COLOUR + TIMESTAMP overlay\n"
gstd-client pipeline_create pipe_3_src \
  "rtspsrc location=${CAM_URL} latency=${LAT} ! \
   rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! \
   clockoverlay halignment=right valignment=bottom shaded-background=true \
                font-desc=\"Sans Bold 24\" color=\"0xffff0000\" ! \
   videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=15 intra-refresh=true ! \
   rtph264pay pt=98 config-interval=1 ! \
   interpipesink name=src_clock sync=false async=false drop=true"

########################################################
# 2. sink pipeline that we will retune at run time
########################################################

echo -e "\n ✳️  pipe_sink : RTSP forwarder (listens to the chosen source)\n"
gstd-client pipeline_create pipe_sink \
  "interpipesrc name=selector listen-to=src_gray \
      is-live=true do-timestamp=true ! \
   rtspclientsink location=${LOCAL_OUT} protocols=tcp latency=0"



########################################################
# 3. PLAY everything
########################################################

for P in pipe_1_src pipe_2_src pipe_3_src pipe_sink; do
    gstd-client pipeline_play $P
done

########################################################
# 4. round-robin switcher
########################################################

echo -e "\n 🔄  Cycling: RAW → GRAY → CLOCK every 3 s  (Ctrl-C to quit)\n"
while true; do
    gstd-client element_set pipe_sink selector listen-to src_gray
    echo -e "   → now GRAYSCALE"
    sleep 3; [[ $STOP -ne 0 ]] && break

    gstd-client element_set pipe_sink selector listen-to src_clock
    echo -e "   → now CLOCK OVERLAY"
    sleep 3; [[ $STOP -ne 0 ]] && break

    gstd-client element_set pipe_sink selector listen-to src_raw
    echo -e "   → now RAW COLOUR"
    sleep 3; [[ $STOP -ne 0 ]] && break
done

########################################################
# 5. tidy up
########################################################

echo -e "\n ⏹  Ctrl-C caught – cleaning up …\n"
for P in pipe_1_src pipe_2_src pipe_3_src pipe_sink; do
    gstd-client pipeline_delete $P
done

echo -e "\n ✅  Finished – all pipelines removed.\n"
