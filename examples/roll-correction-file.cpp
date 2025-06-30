#include "video/RollCorrection.h"
#include "video/CamCap.h"
#include "video/AutoZoomCrop.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// Compilation:
// g++ video_auto_roll_correction.cpp -o video_auto_roll_correction $(pkg-config --cflags --libs opencv4) -L/usr/local/lib \
// -lvideo-stab -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_videoio \
// -lopencv_cudaoptflow -lopencv_cudawarping -lopencv_calib3d -lopencv_cudaimgproc

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_source>\n";
        return 1;
    }

    std::string videoSource = argv[1];
// GStreamer pipeline for low-latency RTSP capture
     vs::CamCap::Parameters params;
    // rtsp://127.0.0.1:8554/test, long.mp4, long_low.m4v, 0
    params.source = videoSource;
    params.threadedQueueMode = true; // Enable threaded mode
    params.colorspace = "";  // Example color conversion
    params.logging = true;          // Show debug logs
    params.timeDelay = 0;           // 1 second warm-up
    params.threadTimeout = 5000;    // 5 seconds read timeout

        vs::CamCap cam(params);
        cam.start();  // Start the capture thread (since threaded mode is true)


        // Get video properties
    double fps = cam.getFrameRate();
        if (fps < 1.0) fps = 30.0;
            std::cout << "Video framerate: " << fps << " FPS" << std::endl;
            int delayMs = static_cast<int>(1000.0 / fps);

        // Create OpenCV windows
        const int windowWidth = 640;
        const int windowHeight = 360;

        cv::namedWindow("Raw", cv::WINDOW_NORMAL);
        cv::resizeWindow("Raw", windowWidth, windowHeight);

        cv::namedWindow("Auto Roll Corrected", cv::WINDOW_NORMAL);
        cv::resizeWindow("Auto Roll Corrected", windowWidth, windowHeight);

        cv::namedWindow("finalFrame", cv::WINDOW_NORMAL);
        cv::resizeWindow("finalFrame", windowWidth, windowHeight);

        while (true) {
            cv::Mat frame = cam.read();
            if(!frame.empty()) {
               cv::imshow("Raw", frame);
            }

            cv::imshow("Raw", frame);

            // Auto-detect and correct roll
            cv::Mat correctedFrame = vs::RollCorrection::autoCorrectRoll(frame);
            if (!correctedFrame.empty()) {
                cv::imshow("Auto Roll Corrected", correctedFrame);

                cv::Mat finalFrame = vs::AutoZoomCrop::autoZoomCrop(correctedFrame, 0.05); 
                // 0.05 => 5% margin, tweak as you like
                cv::imshow("finalFrame", finalFrame);

            }

            if (cv::waitKey(1) == 27) break; // Press ESC to exit
        }

    cv::destroyAllWindows();
    return 0;
}
