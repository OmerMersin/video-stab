#ifndef CAMCAP_H
#define CAMCAP_H

#include <opencv2/opencv.hpp>
#ifdef HAVE_OPENCV_CUDACODEC
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>

/**
 * CamCap C++ version that mimics the logic of camgear.py
 * including optional Threaded Queue mode and CUDA-based color conversion.
 */
namespace vs {
    
    class CamCap {
    public:
        struct Parameters {
            std::string source = "0";  ///< For cameras, e.g. "0", or file path, or RTSP URL, etc.
            bool streamMode = false;   ///< (Optional) If you plan to handle special streaming logic
            int backend = 0;           ///< cv::VideoCapture API backend, e.g. cv::CAP_FFMPEG, etc.
            std::string colorspace;    ///< e.g. "BGR2GRAY", "BGR2HSV", "BGR2YUV", etc.
            bool logging = false;      ///< Enable/disable logging output
            int timeDelay = 0;         ///< Delay in seconds before reading frames (camera warm-up)
            bool threadedQueueMode = true; ///< If true, a separate thread pushes frames into a queue
            int queueSize = 96;        ///< Max queue size if using threaded mode
            int threadTimeout = 300000; ///< Time in ms to wait for a frame in read(); <=0 means no timeout
        };

        explicit CamCap(const Parameters& params);
        ~CamCap();

        void start();           ///< Start thread if threadedQueueMode is true
        cv::Mat read();         ///< Get the next frame (from queue if threaded mode, else direct)
        void stop();            ///< Stop capturing and release resources

        double getFrameRate() const { return framerate; }

    private:
        void updateLoop();      ///< Threaded loop that grabs frames from cap
        void cudaConvertColor(cv::Mat& frame);

        // Internal variables
        cv::VideoCapture cap;
        std::atomic<bool> terminate{false};
        std::atomic<bool> isRunning{false};

        // Thread and Queue
        std::thread captureThread;
        std::mutex queueMutex;
        std::condition_variable queueCondition;
        std::queue<cv::Mat> frameQueue;

        // Frame-related
        cv::Mat currentFrame;
        double framerate = 0.0;
        int colorConversionCode = -1;

        // Parameters
        Parameters params;

    #ifdef HAVE_OPENCV_CUDACODEC
        cv::cuda::Stream cudaStream;  ///< For asynchronous CUDA operations
    #endif
    };

    #endif // CAMCAP_H
}