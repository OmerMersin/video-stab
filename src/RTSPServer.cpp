#include "video/RTSPServer.h"

#include <gst/app/gstappsrc.h>
#include <iostream>
#include <chrono>
#include <cstring>

RTSPServer::RTSPServer()
    : mainLoop(nullptr),
      rtspServer(nullptr),
      mounts(nullptr),
      factory(nullptr),
      appsrc(nullptr)
{
    // Initialize GStreamer (safe to call multiple times, but typically do it once)
    gst_init(nullptr, nullptr);
}

RTSPServer::~RTSPServer()
{
    // Stop main loop if running
    if (mainLoop) {
        g_main_loop_quit(mainLoop);
        if (mainLoopThread.joinable()) {
            mainLoopThread.join();
        }
        mainLoop = nullptr;
    }

    // Cleanup RTSP server
    if (rtspServer) {
        g_object_unref(rtspServer);
        rtspServer = nullptr;
    }
}

bool RTSPServer::startServer(int port, const std::string &mountPoint)
{
    // 1) Create a GMainLoop
    mainLoop = g_main_loop_new(nullptr, FALSE);
    if (!mainLoop) {
        std::cerr << "Failed to create GMainLoop." << std::endl;
        return false;
    }

    // 2) Create RTSP server instance
    rtspServer = gst_rtsp_server_new();
    if (!rtspServer) {
        std::cerr << "Failed to create GstRTSPServer." << std::endl;
        return false;
    }

    // Set server port
    gchar *portStr = g_strdup_printf("%d", port);
    g_object_set(rtspServer, "service", portStr, NULL);
    g_free(portStr);

    // 3) Get RTSP mount points
    mounts = gst_rtsp_server_get_mount_points(rtspServer);
    if (!mounts) {
        std::cerr << "Failed to get mount points from RTSP server." << std::endl;
        return false;
    }

    // 4) Create a media factory
    factory = gst_rtsp_media_factory_new();
    if (!factory) {
        std::cerr << "Failed to create GstRTSPMediaFactory." << std::endl;
        return false;
    }

    // 5) Set the pipeline launch description
    // Adjust width/height/framerate/bitrate as needed
    gst_rtsp_media_factory_set_launch(factory,
        "( appsrc name=mysrc format=time "
        "   caps=video/x-raw,format=BGR,width=640,height=360,framerate=30/1 "
        "! videoconvert "
        "! video/x-raw,format=I420 "
        "! x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast "
        "! rtph264pay name=pay0 pt=96 )"
    );

    // Make the factory shared => multiple clients can view simultaneously
    gst_rtsp_media_factory_set_shared(factory, TRUE);

    // 6) Connect to "media-configure" so we can store a handle to `appsrc`
    g_signal_connect(factory, "media-configure",
                     G_CALLBACK(onMediaConfigureCallback), this);

    // 7) Attach the factory to desired mount point (e.g. "/test")
    gst_rtsp_mount_points_add_factory(mounts, mountPoint.c_str(), factory);

    // 8) mounts is now owned by the server; unref our local reference
    g_object_unref(mounts);

    // 9) Attach server to default main context
    if (gst_rtsp_server_attach(rtspServer, NULL) == 0) {
        std::cerr << "Failed to attach RTSP server to the main loop." << std::endl;
        return false;
    }

    // 10) Start the main loop in a separate thread
    mainLoopThread = std::thread([this]() {
        g_main_loop_run(mainLoop); // This blocks until g_main_loop_quit()
    });

    std::cout << "RTSP Server started at rtsp://127.0.0.1:" << port << mountPoint << std::endl;
    return true;
}

// --- STATIC CALLBACK ---
void RTSPServer::onMediaConfigureCallback(GstRTSPMediaFactory *factory,
                                          GstRTSPMedia        *media,
                                          gpointer             user_data)
{
    RTSPServer *server = static_cast<RTSPServer*>(user_data);
    server->onMediaConfigure(media);
}

// --- INSTANCE METHOD ---
// Called each time a client requests the media (the pipeline is built)
void RTSPServer::onMediaConfigure(GstRTSPMedia *media)
{
    // 1) Retrieve the pipeline element for this media
    GstElement *element = gst_rtsp_media_get_element(media);

    // 2) Find appsrc by name
    GstElement *src = gst_bin_get_by_name_recurse_up(GST_BIN(element), "mysrc");
    if (!src) {
        std::cerr << "Failed to get appsrc element from media." << std::endl;
        gst_object_unref(element);
        return;
    }

    // 3) Store pointer to use in pushFrame()
    appsrc = src;

    // 4) Optional: configure appsrc
    gst_app_src_set_stream_type(GST_APP_SRC(appsrc), GST_APP_STREAM_TYPE_STREAM);

    // Freed the pipeline reference
    gst_object_unref(element);
}

// --- PUSH FRAMES INTO GSTREAMER ---
void RTSPServer::pushFrame(const cv::Mat &frame)
{
    // 1) Check if appsrc is ready
    if (!appsrc) {
        // Happens if no client has connected yet. We can still keep pushing, or skip.
        std::cerr << "appsrc not initialized yet (no clients?)." << std::endl;
        return;
    }

    // 2) Ensure the input frame matches the pipeline caps:
    //    - BGR
    //    - 640x360 (or whatever you set in the pipeline)
    //    - 30 fps
    // If your frame is a different size or format, you must convert it here.
    // For demonstration, assume it's already BGR and correct size.

    // 3) Create a new GstBuffer of the correct size
    const size_t size = frame.total() * frame.elemSize(); // e.g. width*height*3
    GstBuffer *buffer = gst_buffer_new_allocate(nullptr, size, nullptr);

    // 4) Map buffer for writing
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);

    // 5) Copy Mat data into GStreamer buffer
    std::memcpy(map.data, frame.data, size);

    gst_buffer_unmap(buffer, &map);

    // 6) Timestamp your buffers so downstream elements have timing info.
    //    For simplicity, we keep an increasing frameCount. 30 fps => PTS increments 1/30 sec.
    static guint64 frameCount = 0;
    GST_BUFFER_PTS(buffer)      = gst_util_uint64_scale(frameCount++, GST_SECOND, 30);  // 30 fps
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, 30);

    // 7) Push the buffer into appsrc
    GstFlowReturn ret;
    g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);

    // 8) Release our reference
    gst_buffer_unref(buffer);

    if (ret != GST_FLOW_OK) {
        std::cerr << "Warning: push-buffer returned " << ret << std::endl;
    }
}
