#include "video/Stabilizer.h"
#include "video/CamCap.h"

#include <opencv2/opencv.hpp>
#include <iostream>

// # g++ file-capture.cpp -o file-capture $(pkg-config --cflags --libs opencv4) -L/usr/local/lib \
// # -lvideo-stab -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_videoio \
// # -lopencv_cudaoptflow -lopencv_cudawarping -lopencv_calib3d -lopencv_cudaimgproc

int main() {
    vs::CamCap::Parameters params;
    // rtsp://127.0.0.1:8554/test, long.mp4, long_low.m4v, 0
    params.source = "long_low2.m4v";
    params.threadedQueueMode = true; // Enable threaded mode
    params.colorspace = "";  // Example color conversion
    params.logging = true;          // Show debug logs
    params.timeDelay = 0;           // 1 second warm-up
    params.threadTimeout = 5000;    // 5 seconds read timeout

    // 2. Set up Stabilizer
    vs::Stabilizer::Parameters stabParams;
    stabParams.smoothingRadius = 20; 
    stabParams.borderType = "reflect"; 
    stabParams.borderSize = 0;       // or e.g. 50 for wide border
    stabParams.cropNZoom = false;    // or true if you want that
    stabParams.logging = true;
    stabParams.useCuda = true;      // set to true if you compiled with CUDA
    vs::Stabilizer stab(stabParams);

    // Set fixed window sizes
    const int windowWidth = 640; // Desired fixed width
    const int windowHeight = 360; // Desired fixed height

    // Create windows and set their sizes
    cv::namedWindow("Raw", cv::WINDOW_NORMAL);
    cv::resizeWindow("Raw", windowWidth, windowHeight);

    cv::namedWindow("Stabilized", cv::WINDOW_NORMAL);
    cv::resizeWindow("Stabilized", windowWidth, windowHeight);

    try {
        vs::CamCap cam(params);
        cam.start();  // Start the capture thread (since threaded mode is true)

    // 3. Fetch the framerate reported by the source
    double fps = cam.getFrameRate();
    if (fps < 1.0) {
        // Fallback if the video doesn't report a real FPS
        fps = 30.0;
    }
    std::cout << "Video framerate: " << fps << " FPS" << std::endl;

    // 4. Calculate the delay in milliseconds for each frame
    int delayMs = static_cast<int>(1000.0 / fps);


        while(true) {
            cv::Mat frame = cam.read();
            if(!frame.empty()) {
               cv::imshow("Raw", frame);
            }
            
            cv::Mat stabilized = stab.stabilize(frame);
            if(!stabilized.empty()) {
                cv::imshow("Stabilized", stabilized);
            }

            if(cv::waitKey(delayMs) == 27) {
                // Press ESC to exit
                break;
            }
        }
        cam.stop();
    }
    catch(const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
    }

    return 0;
}