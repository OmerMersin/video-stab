#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // ▶️ First pipeline: produces H264 and writes to interpipesink
    GstElement *src_pipeline = gst_parse_launch(
    "rtspsrc name=src location=rtsp://192.168.144.119:554 latency=0 ! "
    "rtph264depay ! h264parse ! interpipesink name=to_output sync=false async=false",
    nullptr);

    GstElement *sink_pipeline = gst_parse_launch(
    "interpipesrc name=from_input listen-to=to_output is-live=true do-timestamp=true ! "
    "rtspclientsink location=rtsp://192.168.144.150:8554/forwarded",
    nullptr);

    // 🧠 Force-push caps from the interpipesink side (like gstd does internally)
    GstElement *sink = gst_bin_get_by_name(GST_BIN(src_pipeline), "to_output");
    GstCaps *caps = gst_caps_from_string("video/x-h264,stream-format=byte-stream,alignment=au");
    g_object_set(sink, "caps", caps, nullptr);
    gst_caps_unref(caps);
    gst_object_unref(sink);

    // Start both pipelines
    gst_element_set_state(src_pipeline, GST_STATE_PLAYING);
    g_usleep(500000); // wait a bit to let source flow
    gst_element_set_state(sink_pipeline, GST_STATE_PLAYING);

    // Wait until user stops
    GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(loop);

    // Cleanup
    gst_element_set_state(sink_pipeline, GST_STATE_NULL);
    gst_element_set_state(src_pipeline, GST_STATE_NULL);
    gst_object_unref(sink_pipeline);
    gst_object_unref(src_pipeline);
    g_main_loop_unref(loop);

    return 0;
}
