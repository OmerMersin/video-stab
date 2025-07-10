#ifndef CAMCAP_INTERPIPE_H
#define CAMCAP_INTERPIPE_H

#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>

/**
 * CamCapInterpipe - Modified CamCap that works with interpipes
 * Receives frames from processing pipeline and sends processed frames back
 */
namespace vs {
    
    class CamCapInterpipe {
    public:
        struct Parameters {
            std::string interpipeInputName = "processing_out";   // Interpipe to receive from
            std::string interpipeOutputName = "processed_out";   // Interpipe to send to
            int width = 1920;
            int height = 1080;
            int fps = 30;
            bool logging = false;
            int queueSize = 5;
            int threadTimeout = 500;
        };

        explicit CamCapInterpipe(const Parameters& params);
        ~CamCapInterpipe();

        bool initialize();
        void start();
        cv::Mat read();
        void write(const cv::Mat& frame);
        void stop();
        bool isHealthy() const;

        double getFrameRate() const { return static_cast<double>(params.fps); }
        double getWidth() const { return static_cast<double>(params.width); }
        double getHeight() const { return static_cast<double>(params.height); }

    private:
        void inputLoop();
        void outputLoop();
        
        static GstFlowReturn newSampleCallback(GstAppSink* sink, gpointer userData);
        static gboolean busCallback(GstBus* bus, GstMessage* message, gpointer userData);

        // GStreamer elements
        GstElement* inputPipeline;
        GstElement* outputPipeline;
        GstAppSink* appSink;
        GstAppSrc* appSrc;
        
        // Threading
        std::atomic<bool> terminate{false};
        std::atomic<bool> isRunning{false};
        std::thread inputThread;
        std::thread outputThread;
        
        // Frame queues
        std::mutex inputQueueMutex;
        std::condition_variable inputQueueCondition;
        std::queue<cv::Mat> inputFrameQueue;
        
        std::mutex outputQueueMutex;
        std::condition_variable outputQueueCondition;
        std::queue<cv::Mat> outputFrameQueue;
        
        Parameters params;
        std::atomic<bool> initialized{false};
    };

} // namespace vs

#endif // CAMCAP_INTERPIPE_H
