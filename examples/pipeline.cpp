#include "video/RollCorrection.h"
#include "video/CamCap.h"
#include "video/AutoZoomCrop.h"
#include "video/Stabilizer.h"
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

    vs::Stabilizer::Parameters stabParams;
    stabParams.smoothingRadius = 20; 
    stabParams.borderType = "replicate"; 
    stabParams.borderSize = 0;
    stabParams.cropNZoom = false;
    stabParams.logging = true;
    stabParams.useCuda = true;
    vs::Stabilizer stab(stabParams);

    vs::CamCap::Parameters params;
    params.source = videoSource;
    params.threadedQueueMode = true;
    params.colorspace = "";
    params.logging = true;
    params.timeDelay = 0;
    params.threadTimeout = 5000;

    vs::CamCap cam(params);
    cam.start();

    double fps = cam.getFrameRate();
    if (fps < 1.0) fps = 30.0;
        
    std::cout << "Video framerate: " << fps << " FPS" << std::endl;
    int delayMs = static_cast<int>(1000.0 / fps);

    const int windowWidth = 640;
    const int windowHeight = 360;

    cv::namedWindow("Raw", cv::WINDOW_NORMAL);
    cv::resizeWindow("Raw", windowWidth, windowHeight);

    cv::namedWindow("Stabilized and Roll Corrected", cv::WINDOW_NORMAL);
    cv::resizeWindow("Stabilized and Roll Corrected", windowWidth, windowHeight);

    while (true) {
        cv::Mat frame = cam.read();
        if(!frame.empty()) {
            cv::imshow("Raw", frame);
        }

        cv::Mat correctedFrame = vs::RollCorrection::autoCorrectRoll(frame);

        cv::Mat stabilized = stab.stabilize(correctedFrame);
        if(!stabilized.empty()) {
            cv::imshow("Stabilized and Roll Corrected", stabilized);
        }
        if (cv::waitKey(1) == 27) break;
    }

    cv::destroyAllWindows();
    return 0;
}
