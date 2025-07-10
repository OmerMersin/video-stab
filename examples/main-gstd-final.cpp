#include "video/RollCorrection.h"
#include "video/CamCap.h"
#include "video/AutoZoomCrop.h"
#include "video/Stabilizer.h"
#include "video/Mode.h"
#include "video/Enhancer.h"
#include "video/RTSPServer.h"
#include "video/DeepStreamTracker.h"
#include "video/TcpReciever.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <X11/Xlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <cstdio>
#include <signal.h>
#include <sstream>
#include <sys/select.h>
#include <unistd.h>
#include <memory>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <future>

// Global variable for signal handling
volatile sig_atomic_t stopRequested = 0;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down gracefully..." << std::endl;
    stopRequested = 1;
}

// Pure GStreamer Pipeline Manager (no GSTD dependency)
// This version uses pure GStreamer with interpipesrc/interpipesink for seamless switching
class GStreamerPipelineManager {
private:
    std::string sourceAddress;
    std::string outputAddress;
    int frameWidth, frameHeight;
    double fps;
    int bitrate;
    bool pipelinesInitialized;
    
    // Dual pipeline approach for true passthrough
    GstElement* passthroughPipeline;  // Direct H.265 forwarding
    GstElement* processingPipeline;   // OpenCV frame processing
    GstElement* persistentPipeline;   // Currently active pipeline
    GstAppSrc* persistentAppsrc;
    bool streamInitialized;
    uint64_t frameCounter;
    
    bool isCurrentlyPassthrough;
    
    // GStreamer callback for bus messages
    static gboolean busCallback(GstBus* bus, GstMessage* msg, gpointer data) {
        GStreamerPipelineManager* manager = static_cast<GStreamerPipelineManager*>(data);
        
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError* err;
                gchar* debug;
                gst_message_parse_error(msg, &err, &debug);
                std::cerr << "GStreamer Error: " << err->message << std::endl;
                if (debug) {
                    std::cerr << "Debug info: " << debug << std::endl;
                }
                g_error_free(err);
                g_free(debug);
                break;
            }
            case GST_MESSAGE_WARNING: {
                GError* err;
                gchar* debug;
                gst_message_parse_warning(msg, &err, &debug);
                std::cerr << "GStreamer Warning: " << err->message << std::endl;
                if (debug) {
                    std::cerr << "Debug info: " << debug << std::endl;
                }
                g_error_free(err);
                g_free(debug);
                break;
            }
            case GST_MESSAGE_EOS:
                std::cout << "End-of-stream reached" << std::endl;
                break;
            case GST_MESSAGE_STATE_CHANGED: {
                if (GST_MESSAGE_SRC(msg) == GST_OBJECT(manager->persistentPipeline)) {
                    GstState old_state, new_state, pending_state;
                    gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
                    std::cout << "Pipeline state changed from " << gst_element_state_get_name(old_state) 
                              << " to " << gst_element_state_get_name(new_state) << std::endl;
                }
                break;
            }
            default:
                break;
        }
        return TRUE;
    }
    
public:
    GStreamerPipelineManager(const std::string& source = "rtsp://192.168.144.119:554",
                           const std::string& output = "rtsp://192.168.144.150:8554/forwarded", 
                           int bitrate = 4000000) 
        : sourceAddress(source), outputAddress(output), bitrate(bitrate), pipelinesInitialized(false),
          passthroughPipeline(nullptr), processingPipeline(nullptr), persistentPipeline(nullptr), 
          persistentAppsrc(nullptr), streamInitialized(false), frameCounter(0), isCurrentlyPassthrough(true) {
        
        // Initialize GStreamer
        if (!gst_is_initialized()) {
            gst_init(nullptr, nullptr);
        }
    }
    
    bool initialize(int width, int height, double framerate) {
        frameWidth = width;
        frameHeight = height;
        fps = framerate;
        
        if (streamInitialized) return true;
        
        // Calculate appropriate bitrate based on resolution
        int calculatedBitrate = (frameWidth * frameHeight * fps * 0.1) / 1000; // In Kbps
        if (calculatedBitrate < 800) calculatedBitrate = 800; // Minimum bitrate floor
        if (calculatedBitrate > 8000) calculatedBitrate = 8000; // Maximum bitrate ceiling
        bitrate = calculatedBitrate;
        
        // Create PASSTHROUGH pipeline (direct H.265 forwarding with hardware decoding/software encoding)
        std::string passthroughPipelineStr = 
            "rtspsrc name=src location=" + sourceAddress + " latency=0 ! "
            "rtph265depay ! h265parse ! "
            "rtspclientsink location=" + outputAddress + " protocols=tcp";
        
        std::cout << "Creating passthrough pipeline: " << passthroughPipelineStr << std::endl;
        passthroughPipeline = gst_parse_launch(passthroughPipelineStr.c_str(), nullptr);
        if (!passthroughPipeline) {
            std::cerr << "Failed to create passthrough pipeline" << std::endl;
            return false;
        }
        
        // Create PROCESSING pipeline (OpenCV frame processing with software encoding)
        std::string processingPipelineStr = 
            "appsrc name=src is-live=true format=time block=false max-latency=0 "
            "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) +
            ",height=" + std::to_string(frameHeight) +
            ",framerate=" + std::to_string(static_cast<int>(fps)) + "/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            "x264enc speed-preset=ultrafast tune=zerolatency bitrate=" + std::to_string(calculatedBitrate) + 
            " key-int-max=15 ! "
            "h264parse ! "
            "rtspclientsink location=" + outputAddress + " protocols=tcp latency=0";
        
        std::cout << "Creating processing pipeline: " << processingPipelineStr << std::endl;
        processingPipeline = gst_parse_launch(processingPipelineStr.c_str(), nullptr);
        if (!processingPipeline) {
            std::cerr << "Failed to create processing pipeline" << std::endl;
            return false;
        }
        
        // Get appsrc from processing pipeline
        persistentAppsrc = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(processingPipeline), "src"));
        if (!persistentAppsrc) {
            std::cerr << "Failed to get appsrc element" << std::endl;
            return false;
        }
        
        // Set appsrc properties
        g_object_set(persistentAppsrc,
            "is-live", TRUE,
            "block", FALSE,
            "format", GST_FORMAT_TIME,
            "max-latency", G_GINT64_CONSTANT(0),
            "do-timestamp", TRUE,
            NULL);
        
        // Set up bus message handling for both pipelines
        GstBus* passthroughBus = gst_pipeline_get_bus(GST_PIPELINE(passthroughPipeline));
        gst_bus_add_watch(passthroughBus, busCallback, this);
        gst_object_unref(passthroughBus);
        
        GstBus* processingBus = gst_pipeline_get_bus(GST_PIPELINE(processingPipeline));
        gst_bus_add_watch(processingBus, busCallback, this);
        gst_object_unref(processingBus);
        
        // Start in passthrough mode by default
        std::cout << "Starting passthrough pipeline..." << std::endl;
        if (gst_element_set_state(passthroughPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start passthrough pipeline" << std::endl;
            return false;
        }
        
        // Wait for passthrough pipeline to start
        gst_element_get_state(passthroughPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        persistentPipeline = passthroughPipeline;  // Currently active pipeline
        pipelinesInitialized = true;
        streamInitialized = true;
        frameCounter = 0;
        isCurrentlyPassthrough = true;
        
        std::cout << "Dual pipeline system initialized successfully" << std::endl;
        std::cout << "Source: " << sourceAddress << std::endl;
        std::cout << "Output: " << outputAddress << std::endl;
        std::cout << "Bitrate: " << calculatedBitrate << " Kbps" << std::endl;
        std::cout << "Started in PASSTHROUGH mode (direct H.265 forwarding)" << std::endl;
        return true;
    }
    
    bool switchToPassthrough() {
        if (!pipelinesInitialized) {
            std::cerr << "Pipeline not initialized" << std::endl;
            return false;
        }
        
        if (isCurrentlyPassthrough) {
            std::cout << "Already in passthrough mode" << std::endl;
            return true;
        }
        
        // Stop processing pipeline
        std::cout << "Stopping processing pipeline..." << std::endl;
        gst_element_set_state(processingPipeline, GST_STATE_NULL);
        gst_element_get_state(processingPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        // Start passthrough pipeline
        std::cout << "Starting passthrough pipeline..." << std::endl;
        if (gst_element_set_state(passthroughPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start passthrough pipeline" << std::endl;
            return false;
        }
        
        // Wait for passthrough pipeline to start
        gst_element_get_state(passthroughPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        persistentPipeline = passthroughPipeline;
        isCurrentlyPassthrough = true;
        
        std::cout << "Switched to PASSTHROUGH mode - direct H.265 forwarding" << std::endl;
        return true;
    }
    
    bool switchToProcessing() {
        if (!pipelinesInitialized) {
            std::cerr << "Pipeline not initialized" << std::endl;
            return false;
        }
        
        if (!isCurrentlyPassthrough) {
            std::cout << "Already in processing mode" << std::endl;
            return true;
        }
        
        // Stop passthrough pipeline
        std::cout << "Stopping passthrough pipeline..." << std::endl;
        gst_element_set_state(passthroughPipeline, GST_STATE_NULL);
        gst_element_get_state(passthroughPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        // Start processing pipeline
        std::cout << "Starting processing pipeline..." << std::endl;
        if (gst_element_set_state(processingPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start processing pipeline" << std::endl;
            return false;
        }
        
        // Wait for processing pipeline to start
        gst_element_get_state(processingPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        persistentPipeline = processingPipeline;
        isCurrentlyPassthrough = false;
        
        std::cout << "Switched to PROCESSING mode - will process frames from OpenCV" << std::endl;
        return true;
    }
    
    bool pushFrame(const cv::Mat& frame) {
        if (!pipelinesInitialized) {
            std::cerr << "Pipeline not initialized, cannot push frame" << std::endl;
            return false;
        }
        
        if (isCurrentlyPassthrough) {
            // In passthrough mode, we don't push OpenCV frames
            // The passthrough pipeline handles the stream directly
            return true;
        }
        
        if (frame.empty()) {
            std::cerr << "Received empty frame to push" << std::endl;
            return false;
        }
        
        // Verify processing pipeline state before pushing
        GstState state;
        GstStateChangeReturn ret = gst_element_get_state(processingPipeline, &state, nullptr, 0);
        if (state != GST_STATE_PLAYING) {
            std::cerr << "Processing pipeline not in PLAYING state: " << gst_element_state_get_name(state) << std::endl;
            return false;
        }
        
        // PROCESSING MODE: Full validation and format conversion
        cv::Mat outputFrame;
        if (frame.cols != frameWidth || frame.rows != frameHeight) {
            cv::resize(frame, outputFrame, cv::Size(frameWidth, frameHeight), 0, 0, cv::INTER_LANCZOS4);
        } else {
            // Ensure frame is continuous in memory
            if (!frame.isContinuous()) {
                frame.copyTo(outputFrame);
            } else {
                outputFrame = frame.clone();
            }
        }
        
        // Ensure BGR format (GStreamer expects this)
        if (outputFrame.channels() != 3) {
            cv::cvtColor(outputFrame, outputFrame, cv::COLOR_GRAY2BGR);
        }
        
        // Create GStreamer buffer with proper size
        size_t bufferSize = outputFrame.total() * outputFrame.elemSize();
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bufferSize, nullptr);
        
        if (!buffer) {
            std::cerr << "Failed to allocate GStreamer buffer" << std::endl;
            return false;
        }
        
        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            std::cerr << "Failed to map GStreamer buffer" << std::endl;
            gst_buffer_unref(buffer);
            return false;
        }
        
        memcpy(map.data, outputFrame.data, bufferSize);
        gst_buffer_unmap(buffer, &map);
        
        // Set proper timestamp (like main-gst.cpp)
        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frameCounter++, GST_SECOND, fps);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, fps);
        
        // Push buffer to pipeline
        GstFlowReturn flowRet = gst_app_src_push_buffer(persistentAppsrc, buffer);
        
        if (flowRet != GST_FLOW_OK) {
            std::cerr << "Failed to push frame to pipeline: " << gst_flow_get_name(flowRet) << std::endl;
            return false;
        }
        
        // Debug info every 60 frames
        if (frameCounter % 60 == 0) {
            std::cout << "Successfully pushed frame #" << frameCounter << " to pipeline" << std::endl;
        }
        
        return true;
    }
    
    void printStatus() {
        if (!pipelinesInitialized) {
            std::cout << "Pipeline Manager Status: Not initialized" << std::endl;
            return;
        }
        
        std::cout << "Pipeline Manager Status:" << std::endl;
        std::cout << "  - Source Address: " << sourceAddress << std::endl;
        std::cout << "  - Output Address: " << outputAddress << std::endl;
        std::cout << "  - Resolution: " << frameWidth << "x" << frameHeight << std::endl;
        std::cout << "  - FPS: " << fps << std::endl;
        std::cout << "  - Bitrate: " << bitrate << " Kbps" << std::endl;
        std::cout << "  - Current Mode: " << (isCurrentlyPassthrough ? "Passthrough (Direct H.265)" : "Processing (Enhanced)") << std::endl;
        std::cout << "  - Frame Counter: " << frameCounter << std::endl;
        
        // Print pipeline states for debugging
        GstState passthroughState, processingState;
        if (passthroughPipeline) {
            gst_element_get_state(passthroughPipeline, &passthroughState, nullptr, 0);
            std::cout << "  - Passthrough Pipeline: " << gst_element_state_get_name(passthroughState) << std::endl;
        }
        
        if (processingPipeline) {
            gst_element_get_state(processingPipeline, &processingState, nullptr, 0);
            std::cout << "  - Processing Pipeline: " << gst_element_state_get_name(processingState) << std::endl;
        }
    }
    
    void cleanup() {
        if (passthroughPipeline) {
            gst_element_set_state(passthroughPipeline, GST_STATE_NULL);
            gst_object_unref(passthroughPipeline);
            passthroughPipeline = nullptr;
        }
        
        if (processingPipeline) {
            gst_element_set_state(processingPipeline, GST_STATE_NULL);
            gst_object_unref(processingPipeline);
            processingPipeline = nullptr;
        }
        
        if (persistentAppsrc) {
            gst_object_unref(persistentAppsrc);
            persistentAppsrc = nullptr;
        }
        
        persistentPipeline = nullptr;  // Just a pointer, don't unref
        pipelinesInitialized = false;
    }
    
    ~GStreamerPipelineManager() {
        cleanup();
    }
};

// Function to read configurations from a YAML file
bool readConfig(
    const std::string& filename, 
    std::string& videoSource, 
    vs::Mode::Parameters& runParams, 
    vs::Enhancer::Parameters& enhancerParams,
    vs::RollCorrection::Parameters& rollParams,
    vs::Stabilizer::Parameters& stabParams, 
    vs::CamCap::Parameters& camParams,
    vs::DeepStreamTracker::Parameters& trackerParams
) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return false;
    }

    fs["video_source"] >> videoSource;

    // Read Mode Parameters
    cv::FileNode modeNode = fs["mode"];
    if (!modeNode.empty()) {
        modeNode["width"] >> runParams.width;
        modeNode["height"] >> runParams.height;
        modeNode["optimize_fps"] >> runParams.optimizeFps;
        modeNode["use_cuda"] >> runParams.useCuda;
        modeNode["enhancer_enabled"] >> runParams.enhancerEnabled;
        modeNode["roll_correction_enabled"] >> runParams.rollCorrectionEnabled;
        modeNode["stabilizer_enabled"] >> runParams.stabilizationEnabled;
        modeNode["tracker_enabled"] >> runParams.trackerEnabled;
    }

    // Read Enhancer Parameters
    cv::FileNode enhancerNode = fs["enhancer"];
    if (!enhancerNode.empty()) {
        enhancerNode["brightness"] >> enhancerParams.brightness;
        enhancerNode["contrast"] >> enhancerParams.contrast;
        enhancerNode["enable_white_balance"] >> enhancerParams.enableWhiteBalance;
        enhancerNode["wb_strength"] >> enhancerParams.wbStrength;
        enhancerNode["enable_vibrance"] >> enhancerParams.enableVibrance;
        enhancerNode["vibrance_strength"] >> enhancerParams.vibranceStrength;
        enhancerNode["enable_unsharp"] >> enhancerParams.enableUnsharp;
        enhancerNode["sharpness"] >> enhancerParams.sharpness;
        enhancerNode["blur_sigma"] >> enhancerParams.blurSigma;
        enhancerNode["enable_denoise"] >> enhancerParams.enableDenoise;
        enhancerNode["denoise_strength"] >> enhancerParams.denoiseStrength;
        enhancerNode["gamma"] >> enhancerParams.gamma;
        enhancerNode["enable_clahe"] >> enhancerParams.enableClahe;
        enhancerNode["clahe_clip_limit"] >> enhancerParams.claheClipLimit;
        enhancerNode["clahe_tile_grid_size"] >> enhancerParams.claheTileGridSize;
        enhancerNode["use_cuda"] >> enhancerParams.useCuda;
    }

    // Read Roll Correction Parameters
    cv::FileNode rollNode = fs["roll_correction"];
    if (!rollNode.empty()) {
        rollNode["scale_factor"] >> rollParams.scaleFactor;
        rollNode["canny_threshold_low"] >> rollParams.cannyThresholdLow;
        rollNode["canny_threshold_high"] >> rollParams.cannyThresholdHigh;
        rollNode["canny_aperture"] >> rollParams.cannyAperture;
        rollNode["hough_rho"] >> rollParams.houghRho;
        rollNode["hough_theta"] >> rollParams.houghTheta;
        rollNode["hough_threshold"] >> rollParams.houghThreshold;
        rollNode["angle_smoothing_alpha"] >> rollParams.angleSmoothingAlpha;
        rollNode["angle_decay"] >> rollParams.angleDecay;
        rollNode["angle_filter_min"] >> rollParams.angleFilterMin;
        rollNode["angle_filter_max"] >> rollParams.angleFilterMax;
    }

    // Read Stabilizer Parameters
    cv::FileNode stabNode = fs["stabilizer"];
    if (!stabNode.empty()) {
        stabNode["smoothing_radius"] >> stabParams.smoothingRadius;
        stabNode["border_type"] >> stabParams.borderType;
        stabNode["border_size"] >> stabParams.borderSize;
        stabNode["crop_n_zoom"] >> stabParams.cropNZoom;
        stabNode["logging"] >> stabParams.logging;
        stabNode["use_cuda"] >> stabParams.useCuda;
    }

    // Read Camera Parameters
    cv::FileNode camNode = fs["camera"];
    if (!camNode.empty()) {
        camNode["threaded_queue_mode"] >> camParams.threadedQueueMode;
        camNode["colorspace"] >> camParams.colorspace;
        camNode["logging"] >> camParams.logging;
        camNode["time_delay"] >> camParams.timeDelay;
        camNode["thread_timeout"] >> camParams.threadTimeout;
    }

    // Read Tracker Parameters
    cv::FileNode trackerNode = fs["deepstream_tracker"];
    if (!trackerNode.empty()) {
        std::string modelEngine, modelConfigFile, trackerConfigFile;
        int processingWidth = 640, processingHeight = 384;
        
        trackerNode["model_engine"] >> modelEngine;
        trackerNode["model_config_file"] >> modelConfigFile;
        trackerNode["tracker_config_file"] >> trackerConfigFile;
        trackerNode["processing_width"] >> processingWidth;
        trackerNode["processing_height"] >> processingHeight;
        trackerNode["confidence_threshold"] >> trackerParams.confidenceThreshold;
        
        if (!modelEngine.empty())
            trackerParams.modelEngine = modelEngine;
        if (!modelConfigFile.empty())
            trackerParams.modelConfigFile = modelConfigFile;
        if (!trackerConfigFile.empty())
            trackerParams.trackerConfigFile = trackerConfigFile;
        
        trackerParams.processingWidth = processingWidth;
        trackerParams.processingHeight = processingHeight;
        trackerNode["batch_size"] >> trackerParams.batchSize;
        trackerNode["debug_mode"] >> trackerParams.debugMode;
    }

    fs.release();
    return true;
}

// Main program
int main(int argc, char* argv[]) {
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    if (!XInitThreads()) {
        std::cerr << "XInitThreads() failed." << std::endl;
        return 1;
    }

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>\n";
        return 1;
    }
    std::string configFile = argv[1];
    
    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Configuration monitoring
    time_t lastConfigModTime = 0;
    struct stat configStat;
    if (stat(configFile.c_str(), &configStat) == 0) {
        lastConfigModTime = configStat.st_mtime;
    }

    // Default parameters
    std::string videoSource;
    vs::Mode::Parameters runParams;
    vs::Enhancer::Parameters enhancerParams;
    vs::RollCorrection::Parameters rollParams;
    vs::Stabilizer::Parameters stabParams;
    vs::CamCap::Parameters camParams;
    vs::DeepStreamTracker::Parameters trackerParams;
    
    // Set default tracker parameters
    trackerParams.modelEngine = "/opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine";
    trackerParams.modelConfigFile = "/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_infer_primary_resnet18.txt";
    trackerParams.processingWidth = 640;
    trackerParams.processingHeight = 368;
    trackerParams.confidenceThreshold = 0.3;

    // Read the config file
    if (!readConfig(configFile, videoSource, runParams, enhancerParams, rollParams, stabParams, camParams, trackerParams)) {
        return 1;
    }

    std::cout << "Using video source: " << videoSource << std::endl;
    std::cout << "Roll Correction: " << (runParams.rollCorrectionEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Stabilizer: " << (runParams.stabilizationEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Tracker: " << (runParams.trackerEnabled ? "Enabled" : "Disabled") << std::endl;

    // Initialize processing modules
    vs::Stabilizer stab(stabParams);
    
    // Initialize tracker and TCP receiver
    std::unique_ptr<vs::DeepStreamTracker> tracker = std::make_unique<vs::DeepStreamTracker>(trackerParams);
    vs::TcpReciever tcp(5000);
    tcp.start();
    int x = -1, y = -1;  // Tracking coordinates

    // Initialize camera for frame capture in processing mode
    camParams.source = videoSource;
    std::unique_ptr<vs::CamCap> cam = std::make_unique<vs::CamCap>(camParams);
    cam->start();

    // Set frame properties from config
    double fps = 30.0;  // Default fps
    int frameWidth = runParams.width > 0 ? runParams.width : 1920;
    int frameHeight = runParams.height > 0 ? runParams.height : 1080;
    
    std::cout << "Frame dimensions: " << frameWidth << "x" << frameHeight << " @ " << fps << " fps" << std::endl;

    // Determine initial mode
    bool usePassthrough = !runParams.enhancerEnabled && 
                         !runParams.rollCorrectionEnabled && 
                         !runParams.stabilizationEnabled &&
                         !runParams.trackerEnabled;

    // Create and initialize pipeline manager with source and output addresses
    std::string sourceAddress = videoSource;  // Use video source from config
    std::string outputAddress = "rtsp://192.168.144.150:8554/forwarded";  // RTSP output
    int bitrate = std::max(2000000, std::min(8000000, static_cast<int>(frameWidth * frameHeight * fps * 0.1)));
    
    GStreamerPipelineManager pipelineManager(sourceAddress, outputAddress, bitrate);
    
    if (!pipelineManager.initialize(frameWidth, frameHeight, fps)) {
        std::cerr << "Failed to initialize pipeline manager" << std::endl;
        return -1;
    }

    // Set initial mode based on config
    if (usePassthrough) {
        pipelineManager.switchToPassthrough();
        std::cout << "Starting in PASSTHROUGH mode - direct RTSP forwarding" << std::endl;
    } else {
        pipelineManager.switchToProcessing();
        std::cout << "Starting in PROCESSING mode - will process frames when available" << std::endl;
    }

    std::cout << "GStreamer-based seamless switching is active" << std::endl;

    // Create display windows if needed
    const int windowWidth = runParams.width;
    const int windowHeight = runParams.height;
    
    if (!runParams.optimizeFps) {
        cv::namedWindow("Raw", cv::WINDOW_NORMAL);
        cv::resizeWindow("Raw", windowWidth, windowHeight);
        cv::namedWindow("Final", cv::WINDOW_NORMAL);
        cv::resizeWindow("Final", windowWidth, windowHeight);
    }

    // Main processing loop
    int frameCounter = 0;
    while (!stopRequested) {
        // Configuration file monitoring
        if (frameCounter % 30 == 0) {  // Check every 30 frames
            if (stat(configFile.c_str(), &configStat) == 0) {
                if (configStat.st_mtime != lastConfigModTime) {
                    std::cout << "\n=== Configuration file updated, reloading parameters... ===" << std::endl;
                    
                    // Store old values
                    bool oldPassthrough = usePassthrough;
                    
                    if (readConfig(configFile, videoSource, runParams, enhancerParams, rollParams, stabParams, camParams, trackerParams)) {
                        // Update processing mode
                        usePassthrough = !runParams.enhancerEnabled && 
                                       !runParams.rollCorrectionEnabled && 
                                       !runParams.stabilizationEnabled &&
                                       !runParams.trackerEnabled;
                        
                        // Switch mode if needed
                        if (oldPassthrough != usePassthrough) {
                            if (usePassthrough) {
                                pipelineManager.switchToPassthrough();
                                std::cout << "Switched to PASSTHROUGH mode due to config change" << std::endl;
                            } else {
                                pipelineManager.switchToProcessing();
                                std::cout << "Switched to PROCESSING mode due to config change" << std::endl;
                            }
                        }
                        
                        lastConfigModTime = configStat.st_mtime;
                        std::cout << "Configuration reloaded successfully!" << std::endl;
                    } else {
                        std::cerr << "Failed to reload configuration" << std::endl;
                    }
                }
            }
        }

        // In the new architecture, RTSP streams are handled directly by GStreamer
        // We only need to handle processing mode when frames need to be processed
        if (!usePassthrough) {
            // Check camera health
            if (!cam->isHealthy()) {
                std::cerr << "Camera is not healthy, attempting to restart..." << std::endl;
                cam->stop();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                cam->start();
                continue;
            }
            
            // Process frames from camera when in processing mode
            auto startTime = std::chrono::high_resolution_clock::now();
            cv::Mat frame = cam->read();
            
            static int emptyFrameCount = 0;
            if (frame.empty()) {
                emptyFrameCount++;
                if (emptyFrameCount % 30 == 0) {
                    std::cerr << "No frames received from camera, empty frame count: " << emptyFrameCount << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            } else {
                if (emptyFrameCount > 0) {
                    std::cout << "Frame received after " << emptyFrameCount << " empty frames" << std::endl;
                }
                emptyFrameCount = 0;
            }

            // Debug info for first few frames
            if (frameCounter < 5) {
                std::cout << "Processing frame #" << frameCounter << " - Size: " << frame.cols << "x" << frame.rows 
                         << " Channels: " << frame.channels() << std::endl;
            }

            // Frame processing pipeline
            cv::Mat* framePtr = &frame;
            cv::Mat tempFrame1, tempFrame2, tempFrame3;
            
            // Apply processing steps based on configuration
            if (runParams.enhancerEnabled) {
                try {
                    tempFrame1 = vs::Enhancer::enhanceImage(*framePtr, enhancerParams);
                    framePtr = &tempFrame1;
                    if (frameCounter < 5) std::cout << "Applied enhancer" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Enhancer error: " << e.what() << std::endl;
                }
            }

            // Apply Roll Correction
            if (runParams.rollCorrectionEnabled) {
                try {
                    tempFrame2 = vs::RollCorrection::autoCorrectRoll(*framePtr, rollParams);
                    framePtr = &tempFrame2;
                    if (frameCounter < 5) std::cout << "Applied roll correction" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Roll correction error: " << e.what() << std::endl;
                }
            }

            // Apply Stabilization
            if (runParams.stabilizationEnabled) {
                try {
                    tempFrame3 = stab.stabilize(*framePtr);
                    if (!tempFrame3.empty()) {
                        framePtr = &tempFrame3;
                        if (frameCounter < 5) std::cout << "Applied stabilization" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Stabilization error: " << e.what() << std::endl;
                }
            }

            // Apply tracking if enabled
            if (runParams.trackerEnabled) {
                try {
                    // Get tracking coordinates from TCP
                    if (tcp.tryGetLatest(x, y)) {
                        std::cout << "Received tracking coordinates: (" << x << ", " << y << ")" << std::endl;
                    }
                    
                    // Process frame through tracker
                    auto detections = tracker->processFrame(*framePtr);
                    
                    // Draw detections with the selected coordinates
                    cv::Mat trackedFrame;
                    if (x >= 0 && y >= 0) {
                        trackedFrame = tracker->drawDetections(*framePtr, detections, x, y);
                    } else {
                        trackedFrame = tracker->drawDetections(*framePtr, detections);
                    }
                    
                    if (!trackedFrame.empty()) {
                        tempFrame3 = trackedFrame;
                        framePtr = &tempFrame3;
                        if (frameCounter < 5) std::cout << "Applied tracking" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Tracking error: " << e.what() << std::endl;
                }
            }

            // Push processed frame to GStreamer pipeline
            if (!framePtr->empty()) {
                bool success = pipelineManager.pushFrame(*framePtr);
                if (!success) {
                    std::cerr << "Failed to push processed frame to pipeline at frame #" << frameCounter << std::endl;
                } else if (frameCounter < 5) {
                    std::cout << "Successfully pushed frame #" << frameCounter << " to pipeline" << std::endl;
                }
            } else {
                std::cerr << "Processed frame is empty, skipping push to pipeline" << std::endl;
            }

            // Display processed frame occasionally if not optimizing for FPS
            if (!runParams.optimizeFps && frameCounter % 30 == 0) {
                cv::Mat displayFrame;
                if (windowWidth > 0 && windowHeight > 0) {
                    cv::resize(*framePtr, displayFrame, cv::Size(windowWidth, windowHeight));
                    cv::imshow("Final", displayFrame);
                } else {
                    cv::imshow("Final", *framePtr);
                }
            }

            // Measure Processing Time
            auto endTime = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            
            // Print processing time occasionally
            if (frameCounter % 300 == 0) {
                std::cout << "Processing time: " << frameTime << " ms/frame" << std::endl;
                std::cout << "Frame rate: " << (1000.0 / frameTime) << " fps" << std::endl;
            }
        }

        frameCounter++;
        
        // Print performance stats occasionally
        if (frameCounter % 300 == 0) {  // Every 300 frames (~10 seconds at 30fps)
            std::cout << "Frame counter: " << frameCounter << " | Mode: " << 
                      (usePassthrough ? "Passthrough" : "Processing") << std::endl;
            
            pipelineManager.printStatus();
        }

        // Handle keyboard input
        if (!runParams.optimizeFps && frameCounter % 10 == 0) {
            int key = cv::waitKey(1) & 0xFF;
            
            switch (key) {
                case 'p':
                    usePassthrough = true;
                    pipelineManager.switchToPassthrough();
                    std::cout << "Manually switched to PASSTHROUGH mode" << std::endl;
                    break;
                    
                case 'r':
                    usePassthrough = false;
                    pipelineManager.switchToProcessing();
                    std::cout << "Manually switched to PROCESSING mode" << std::endl;
                    break;
                    
                case 's':
                    pipelineManager.printStatus();
                    break;
                    
                case 'q':
                case 27: // ESC key
                    stopRequested = 1;
                    break;
            }
        }
        
        // Small delay to prevent excessive CPU usage
        if (usePassthrough) {
            // In passthrough mode, we don't need tight timing
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            // In processing mode, maintain proper frame timing
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0 / fps)));
        }
    }
    
    std::cout << "Shutting down..." << std::endl;
    
    // Stop camera capture
    if (cam) {
        cam->stop();
    }
    
    // Stop TCP receiver
    tcp.stop();
    
    // Cleanup pipeline manager
    pipelineManager.cleanup();
    
    cv::destroyAllWindows();
    std::cout << "Cleanup complete." << std::endl;
    return 0;
}
