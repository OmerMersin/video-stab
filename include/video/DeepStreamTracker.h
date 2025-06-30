#ifndef DEEPSTREAM_TRACKER_H
#define DEEPSTREAM_TRACKER_H

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <vector>

// Forward declarations for DeepStream types
typedef struct _NvDsObjectMeta NvDsObjectMeta;
typedef struct _NvDsFrameMeta NvDsFrameMeta;

namespace vs {

class DeepStreamTracker {
public:
    struct Parameters {
        // Primary inference parameters
        std::string modelEngine;
        std::string modelConfigFile;
        
        // Tracker parameters
        std::string trackerConfigFile;
        
        // Processing parameters
        int processingWidth;
        int processingHeight;
        int batchSize;
        bool enableLowLatency;
        
        // Debug options
        bool debugMode;
        bool saveDetectionImages;
        std::string saveImagePath;
        
        // Detection thresholds
        float confidenceThreshold;
        
        // Advanced parameters
        int gpuId;
        int maxTrackedObjects;
        
        // Constructor with default values
        Parameters() : 
            modelEngine("/opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine"),
            modelConfigFile("/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_infer_primary_resnet18.txt"),
            trackerConfigFile("/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"),
            processingWidth(640),
            processingHeight(384),
            batchSize(1),
            enableLowLatency(true),
            debugMode(false),
            saveDetectionImages(false),
            saveImagePath("/tmp/detections/"),
            confidenceThreshold(0.5),
            gpuId(0),
            maxTrackedObjects(100)
        {}
    };

    struct Detection {
        int classId;
        float confidence;
        cv::Rect bbox;
        int trackId;
        std::string label;
    };

    DeepStreamTracker(const Parameters& params = Parameters());
    ~DeepStreamTracker();

    // Initialize DeepStream pipeline
    bool initialize();
    
    // Process a frame and get detections
    std::vector<Detection> processFrame(const cv::Mat& frame);
    
    // Draw detections on a frame
cv::Mat drawDetections(const cv::Mat& frame,
                       const std::vector<Detection>& detections,
                       int selX = -1,
                       int selY = -1);


    
    int pickIdAt(int x, int y) const;
    
    // Release resources
    void release();
    
    // Get the last error message
    std::string getLastError() const;

private:
    Parameters params_;
    GstElement* pipeline_ = nullptr;
    std::atomic<bool> initialized_{false};
    std::string lastErrorMessage_;
    
    // Thread-safe queue for frames
    std::queue<cv::Mat> inputFrameQueue_;
    std::mutex queueMutex_;
    std::condition_variable frameCondition_;

        bool isPointInDetection(const Detection& detection, int x, int y, double scaleX, double scaleY) const;

    
    // Thread-safe storage for detections
    std::vector<Detection> latestDetections_;
    mutable std::mutex detectionsMutex_;

    // Processing thread
    std::thread processingThread_;
    std::atomic<bool> stopProcessing_{false};
    
    // Performance monitoring
    std::atomic<int> frameCount_{0};
    std::atomic<double> totalProcessingTime_{0.0};
    std::chrono::time_point<std::chrono::high_resolution_clock> lastReportTime_;
    
    // Create a DeepStream pipeline
    bool createPipeline();
    
    // Process frames in a separate thread
    void processingLoop();
    
    // Convert OpenCV Mat to GstBuffer with proper format
    GstBuffer* matToGstBuffer(const cv::Mat& frame);
    
    // Process metadata and extract detections
    static GstPadProbeReturn metadataProbeCallback(GstPad* pad, GstPadProbeInfo* info, gpointer userData);
    void processMetadata(NvDsFrameMeta* frameMeta);
    
    // Utility functions
    void reportPerformance();
    void setLastError(const std::string& error);
    bool createDirIfNotExists(const std::string& path);
};

} // namespace vs

#endif // DEEPSTREAM_TRACKER_H