#ifndef RTSPSERVER_H
#define RTSPSERVER_H

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <string>
#include <thread>

class RTSPServer {
public:
    RTSPServer();
    ~RTSPServer();

    // Start the RTSP server on given port and mount point, e.g. "8554" and "/test"
    bool startServer(int port, const std::string &mountPoint, int width = 1920, int height = 1080, int fps = 30);

    // Push frames (OpenCV Mat) to GStreamer pipeline
    void pushFrame(const cv::Mat &frame);
    
    // Check if server is ready to accept frames
    bool isReady() const;

private:
    // Called when media is configured (where we find and store "appsrc")
    static void onMediaConfigureCallback(GstRTSPMediaFactory *factory,
                                         GstRTSPMedia        *media,
                                         gpointer             user_data);
    void onMediaConfigure(GstRTSPMedia *media);

private:
    GMainLoop          *mainLoop;
    std::thread         mainLoopThread;
    GstRTSPServer      *rtspServer;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory*factory;

    // Our handle to appsrc (for pushFrame)
    GstElement         *appsrc;
    
    // Frame parameters
    int frameWidth;
    int frameHeight;
    int framerate;
};

#endif // RTSPSERVER_H
