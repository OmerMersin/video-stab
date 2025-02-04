#include "video/stabilizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// #  g++ file-capture.cpp -o file-capture $(pkg-config --cflags --libs opencv4) -L/usr/local/lib -lvideo-stab \
// # -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_videoio \              
// # -lopencv_cudaoptflow -lopencv_cudawarping -lopencv_calib3d -lopencv_cudaimgproc

int main() {
    // GStreamer pipeline for low-latency RTSP capture
    std::string gst_pipeline = "rtspsrc location=rtsp://192.168.10.120:554/stream1 latency=0 ! "
                               "rtph265depay ! avdec_h265 ! videoconvert ! appsink";

    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't open RTSP stream!" << std::endl;
        return -1;
    }

    // 2. Set up Stabilizer
    video::Stabilizer::Parameters stabParams;
    stabParams.smoothingRadius = 5;
    stabParams.borderType = "reflect";
    stabParams.borderSize = 30;
    stabParams.cropNZoom = true;
    stabParams.logging = true;
    stabParams.useCuda = true;
    stabParams.maxCorners = 500;
    stabParams.qualityLevel = 0.01;
    stabParams.minDistance = 3.0;
    video::Stabilizer stab(stabParams);

    // Set fixed window sizes
    const int windowWidth = 640;
    const int windowHeight = 360;

    cv::namedWindow("Raw", cv::WINDOW_NORMAL);
    cv::resizeWindow("Raw", windowWidth, windowHeight);

    cv::namedWindow("Stabilized", cv::WINDOW_NORMAL);
    cv::resizeWindow("Stabilized", windowWidth, windowHeight);

    // Get FPS
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps < 1.0) fps = 30.0;
    std::cout << "Video framerate: " << fps << " FPS" << std::endl;
    int delayMs = static_cast<int>(1000.0 / fps);

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Error: Frame is empty!" << std::endl;
            break;
        }

        cv::imshow("Raw", frame);

        cv::Mat stabilized = stab.stabilize(frame);
        if (!stabilized.empty()) {
            cv::imshow("Stabilized", stabilized);
        }

        if (cv::waitKey(1) == 27) break;  // ESC to exit
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
