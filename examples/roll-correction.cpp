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
    std::string gst_pipeline = "rtspsrc location=rtsp://192.168.10.120:554/stream1 latency=0 ! "
                               "rtph265depay ! avdec_h265 ! videoconvert ! appsink";

    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't open RTSP stream!" << std::endl;
        return -1;
    }

        // Get video properties
        double fps = cap.get(cv::CAP_PROP_FPS);
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

        // cv::namedWindow("finalFrame", cv::WINDOW_NORMAL);
        // cv::resizeWindow("finalFrame", windowWidth, windowHeight);

        while (true) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "Error: Frame is empty!" << std::endl;
                break;
            }

            cv::imshow("Raw", frame);

            // Auto-detect and correct roll
            cv::Mat correctedFrame = vs::RollCorrection::autoCorrectRoll(frame);
            if (!correctedFrame.empty()) {
                cv::imshow("Auto Roll Corrected", correctedFrame);

                // cv::Mat finalFrame = vs::AutoZoomCrop::autoZoomCrop(correctedFrame, 0.05); 
                // // 0.05 => 5% margin, tweak as you like
                // cv::imshow("finalFrame", finalFrame);

            }

            if (cv::waitKey(1) == 27) break; // Press ESC to exit
        }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
