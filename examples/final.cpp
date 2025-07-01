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
#include <memory>  // For std::unique_ptr
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <future>  // For std::async

// Global variable for signal handling
volatile sig_atomic_t stopRequested = 0;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down gracefully..." << std::endl;
    stopRequested = 1;
}

// GStreamer-based passthrough class for efficient RTSP restreaming
class GStreamerPassthrough {
public:  // Make these public so busCallback can access them
    GstElement* pipeline;
    GMainLoop* loop;
    bool isRunning;
    
private:
    std::thread* loopThread;
    std::string sourceUri;
    std::string outputUri;
    
    static gboolean busCallback(GstBus* bus, GstMessage* msg, gpointer data) {
        GStreamerPassthrough* passthrough = static_cast<GStreamerPassthrough*>(data);
        
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError* err;
                gchar* debug;
                gst_message_parse_error(msg, &err, &debug);
                std::cerr << "GStreamer Passthrough Error: " << err->message << std::endl;
                if (debug) {
                    std::cerr << "Debug info: " << debug << std::endl;
                }
                g_error_free(err);
                g_free(debug);
                
                // Set flag to stop instead of calling stop() directly to avoid deadlock
                passthrough->isRunning = false;
                if (passthrough->loop && g_main_loop_is_running(passthrough->loop)) {
                    g_main_loop_quit(passthrough->loop);
                }
                break;
            }
            case GST_MESSAGE_WARNING: {
                GError* err;
                gchar* debug;
                gst_message_parse_warning(msg, &err, &debug);
                std::cout << "GStreamer Passthrough Warning: " << err->message << std::endl;
                if (debug) {
                    std::cout << "Debug info: " << debug << std::endl;
                }
                g_error_free(err);
                g_free(debug);
                break;
            }
            case GST_MESSAGE_INFO: {
                GError* err;
                gchar* debug;
                gst_message_parse_info(msg, &err, &debug);
                std::cout << "GStreamer Passthrough Info: " << err->message << std::endl;
                if (debug) {
                    std::cout << "Debug info: " << debug << std::endl;
                }
                g_error_free(err);
                g_free(debug);
                break;
            }
            case GST_MESSAGE_QOS: {
                // Quality of Service - important for detecting frame drops/glitches
                GstFormat format;
                guint64 processed, dropped;
                gst_message_parse_qos_stats(msg, &format, &processed, &dropped);
                if (dropped > 0) {
                    std::cout << "QoS: Dropped " << dropped << " frames (processed: " << processed << ")" << std::endl;
                }
                break;
            }
            case GST_MESSAGE_LATENCY:
                std::cout << "Latency message received - redistributing latency" << std::endl;
                gst_bin_recalculate_latency(GST_BIN(passthrough->pipeline));
                break;
            case GST_MESSAGE_BUFFERING: {
                gint percent;
                gst_message_parse_buffering(msg, &percent);
                if (percent < 100) {
                    std::cout << "Buffering: " << percent << "%" << std::endl;
                }
                break;
            }
            case GST_MESSAGE_EOS:
                std::cout << "GStreamer Passthrough: End of stream" << std::endl;
                passthrough->isRunning = false;
                if (passthrough->loop && g_main_loop_is_running(passthrough->loop)) {
                    g_main_loop_quit(passthrough->loop);
                }
                break;
            case GST_MESSAGE_STATE_CHANGED: {
                if (GST_MESSAGE_SRC(msg) == GST_OBJECT(passthrough->pipeline)) {
                    GstState oldState, newState, pendingState;
                    gst_message_parse_state_changed(msg, &oldState, &newState, &pendingState);
                    std::cout << "GStreamer Passthrough: State changed from " 
                              << gst_element_state_get_name(oldState) << " to " 
                              << gst_element_state_get_name(newState) << std::endl;
                }
                break;
            }
            default:
                break;
        }
        return TRUE;
    }
    
    void runLoop() {
        g_main_loop_run(loop);
    }
    
public:
    GStreamerPassthrough(const std::string& source, const std::string& output) 
        : pipeline(nullptr), loop(nullptr), loopThread(nullptr), isRunning(false),
          sourceUri(source), outputUri(output) {
        gst_init(nullptr, nullptr);
        
        // Set GStreamer environment for ultra-low latency (matching FFmpeg behavior)
        g_setenv("GST_RTSP_LATENCY", "0", TRUE);
        g_setenv("GST_BUFFER_POOL_MAX_SIZE", "2", TRUE);
        g_setenv("GST_RTP_JITTERBUFFER_LATENCY", "0", TRUE);
    }
    
    ~GStreamerPassthrough() {
        stop();
    }
    
    bool start() {
        if (isRunning) {
            std::cout << "GStreamer Passthrough already running" << std::endl;
            return true;
        }
        
        std::cout << "Starting GStreamer Passthrough..." << std::endl;
        std::cout << "Source: " << sourceUri << std::endl;
        std::cout << "Output: " << outputUri << std::endl;
        
        // Try different pipeline strategies in order of preference
        // Optimized to match FFmpeg's -fflags nobuffer -flags low_delay -c copy behavior
        std::vector<std::string> pipelineStrategies = {
            // Strategy 1: Ultra-minimal H.265 pipeline - closest to FFmpeg's direct copy
            "rtspsrc location=" + sourceUri + " protocols=udp "
            "latency=0 buffer-mode=none drop-on-latency=true "
            "ntp-sync=false do-rtcp=false timeout=2000000 tcp-timeout=2000000 ! "
            "rtph265depay ! "
            "h265parse disable-passthrough=true config-interval=-1 ! "
            "video/x-h265,stream-format=byte-stream,alignment=au ! "
            "rtspclientsink location=" + outputUri + " protocols=tcp "
            "latency=0 async=false sync=false",
            
            // Strategy 2: Ultra-minimal H.264 pipeline - closest to FFmpeg's direct copy
            "rtspsrc location=" + sourceUri + " protocols=udp "
            "latency=0 buffer-mode=none drop-on-latency=true "
            "ntp-sync=false do-rtcp=false timeout=2000000 tcp-timeout=2000000 ! "
            "rtph264depay ! "
            "h264parse disable-passthrough=true config-interval=-1 ! "
            "video/x-h264,stream-format=byte-stream,alignment=au ! "
            "rtspclientsink location=" + outputUri + " protocols=tcp "
            "latency=0 async=false sync=false",
            
            // Strategy 3: No jitter buffer, direct passthrough - matches FFmpeg -fflags nobuffer
            "rtspsrc location=" + sourceUri + " protocols=udp "
            "latency=0 buffer-mode=none drop-on-latency=true "
            "ntp-sync=false do-rtcp=false ! "
            "application/x-rtp ! "
            "rtpjitterbuffer mode=none latency=0 drop-on-latency=true ! "
            "rtph265depay ! identity sync=false ! "
            "rtspclientsink location=" + outputUri + " protocols=tcp "
            "latency=0 async=false sync=false",
            
            // Strategy 4: Auto-detect codec with zero buffering
            "rtspsrc location=" + sourceUri + " protocols=udp "
            "latency=0 buffer-mode=none drop-on-latency=true "
            "ntp-sync=false do-rtcp=false ! "
            "rtpjitterbuffer mode=none latency=0 drop-on-latency=true ! "
            "parsebin ! identity sync=false ! "
            "rtspclientsink location=" + outputUri + " protocols=tcp "
            "latency=0 async=false sync=false",
            
            // Strategy 5: Fallback with minimal processing (UDP+TCP protocols)
            "rtspsrc location=" + sourceUri + " protocols=udp+tcp "
            "latency=0 buffer-mode=slave drop-on-latency=true "
            "ntp-sync=false do-rtcp=false ! "
            "rtpjitterbuffer mode=slave latency=0 drop-on-latency=true ! "
            "parsebin ! identity sync=false ! "
            "rtspclientsink location=" + outputUri + " protocols=tcp "
            "latency=0 async=false sync=false"
        };
        
        GError* error = nullptr;
        bool pipelineCreated = false;
        
        for (size_t i = 0; i < pipelineStrategies.size() && !pipelineCreated; i++) {
            std::cout << "Trying pipeline strategy " << (i + 1) << "..." << std::endl;
            std::cout << "Pipeline: " << pipelineStrategies[i] << std::endl;
            
            if (error) {
                g_error_free(error);
                error = nullptr;
            }
            
            pipeline = gst_parse_launch(pipelineStrategies[i].c_str(), &error);
            if (pipeline && !error) {
                // Apply ultra-low latency settings to individual elements
                setUltraLowLatencyProperties(pipeline);
                pipelineCreated = true;
                std::cout << "Pipeline strategy " << (i + 1) << " created successfully!" << std::endl;
            } else {
                if (pipeline) {
                    gst_object_unref(pipeline);
                    pipeline = nullptr;
                }
                std::cout << "Pipeline strategy " << (i + 1) << " failed";
                if (error) {
                    std::cout << ": " << error->message;
                }
                std::cout << std::endl;
            }
        }
        
        if (!pipelineCreated) {
            std::cerr << "All pipeline strategies failed!" << std::endl;
            if (error) {
                g_error_free(error);
            }
            return false;
        }
        
        // Set up bus callback
        GstBus* bus = gst_element_get_bus(pipeline);
        gst_bus_add_watch(bus, busCallback, this);
        gst_object_unref(bus);
        
        // Create main loop
        loop = g_main_loop_new(nullptr, FALSE);
        
        // Start pipeline
        GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start GStreamer pipeline" << std::endl;
            cleanup();
            return false;
        }
        
        // Start main loop in separate thread
        isRunning = true;
        loopThread = new std::thread(&GStreamerPassthrough::runLoop, this);
        
        std::cout << "GStreamer Passthrough started successfully!" << std::endl;
        return true;
    }
    
    void stop() {
        if (!isRunning) return;
        
        std::cout << "Stopping GStreamer Passthrough..." << std::endl;
        isRunning = false;
        
        // Stop pipeline first
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
        }
        
        // Stop main loop
        if (loop && g_main_loop_is_running(loop)) {
            g_main_loop_quit(loop);
        }
        
        // Wait for loop thread to finish with timeout
        if (loopThread && loopThread->joinable()) {
            // Give it 2 seconds to finish gracefully
            auto future = std::async(std::launch::async, [this]() {
                loopThread->join();
            });
            
            if (future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
                std::cout << "Thread join timeout, detaching..." << std::endl;
                loopThread->detach();
            }
            
            delete loopThread;
            loopThread = nullptr;
        }
        
        cleanup();
        std::cout << "GStreamer Passthrough stopped." << std::endl;
    }
    
    bool isActive() const {
        return isRunning;
    }
    
private:
    void cleanup() {
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
        
        if (loop) {
            g_main_loop_unref(loop);
            loop = nullptr;
        }
    }
    
    // Function to set ultra-low latency properties on pipeline elements
    void setUltraLowLatencyProperties(GstElement* pipeline) {
        GstIterator* iter = gst_bin_iterate_elements(GST_BIN(pipeline));
        GValue item = G_VALUE_INIT;
        
        while (gst_iterator_next(iter, &item) == GST_ITERATOR_OK) {
            GstElement* element = GST_ELEMENT(g_value_get_object(&item));
            const gchar* elementName = gst_element_get_name(element);
            const gchar* factoryName = GST_OBJECT_NAME(gst_element_get_factory(element));
            
            std::cout << "Configuring element: " << factoryName << " (" << elementName << ")" << std::endl;
            
            // Configure rtspsrc for ultra-low latency
            if (g_str_has_prefix(factoryName, "rtspsrc")) {
                g_object_set(element, 
                    "latency", 0,
                    "buffer-mode", 4,  // none
                    "drop-on-latency", TRUE,
                    "ntp-sync", FALSE,
                    "do-rtcp", FALSE,
                    "timeout", G_GUINT64_CONSTANT(2000000),  // 2s
                    "tcp-timeout", G_GUINT64_CONSTANT(2000000),  // 2s
                    "retry", 3,
                    NULL);
            }
            // Configure rtpjitterbuffer for zero buffering
            else if (g_str_has_prefix(factoryName, "rtpjitterbuffer")) {
                g_object_set(element,
                    "latency", 0,
                    "drop-on-latency", TRUE,
                    "mode", 0,  // none/slave
                    NULL);
            }
            // Configure rtspclientsink for immediate output
            else if (g_str_has_prefix(factoryName, "rtspclientsink")) {
                g_object_set(element,
                    "latency", 0,
                    "async", FALSE,
                    "sync", FALSE,
                    NULL);
            }
            // Configure any queue elements for minimal buffering
            else if (g_str_has_prefix(factoryName, "queue")) {
                g_object_set(element,
                    "max-size-buffers", 1,
                    "max-size-bytes", 0,
                    "max-size-time", G_GUINT64_CONSTANT(0),
                    "leaky", 2,  // downstream
                    NULL);
            }
            // Configure depayloaders
            else if (g_str_has_suffix(factoryName, "depay")) {
                // Most depayloaders don't have specific latency settings, but ensure no buffering
                g_object_set(element, "auto-header-extension", FALSE, NULL);
            }
            // Configure parsers for passthrough mode
            else if (g_str_has_suffix(factoryName, "parse")) {
                if (g_str_has_prefix(factoryName, "h264parse") || g_str_has_prefix(factoryName, "h265parse")) {
                    g_object_set(element,
                        "disable-passthrough", TRUE,
                        "config-interval", -1,
                        NULL);
                }
            }
            
            g_value_unset(&item);
        }
        
        gst_iterator_free(iter);
        std::cout << "Ultra-low latency properties applied to all elements" << std::endl;
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

    // --- 2. Enhancer Parameters ---
    cv::FileNode enhancerNode = fs["enhancer"];
    if (!enhancerNode.empty()) {
        // Basic
        enhancerNode["brightness"] >> enhancerParams.brightness;
        enhancerNode["contrast"] >> enhancerParams.contrast;

        // White Balance
        enhancerNode["enable_white_balance"] >> enhancerParams.enableWhiteBalance;
        enhancerNode["wb_strength"] >> enhancerParams.wbStrength;

        // Vibrance
        enhancerNode["enable_vibrance"] >> enhancerParams.enableVibrance;
        enhancerNode["vibrance_strength"] >> enhancerParams.vibranceStrength;

        // Unsharp (sharpening)
        enhancerNode["enable_unsharp"] >> enhancerParams.enableUnsharp;
        enhancerNode["sharpness"] >> enhancerParams.sharpness;
        enhancerNode["blur_sigma"] >> enhancerParams.blurSigma;

        // Denoise
        enhancerNode["enable_denoise"] >> enhancerParams.enableDenoise;
        enhancerNode["denoise_strength"] >> enhancerParams.denoiseStrength;

        // Gamma
        enhancerNode["gamma"] >> enhancerParams.gamma;

        enhancerNode["enable_clahe"] >> enhancerParams.enableClahe;
        enhancerNode["clahe_clip_limit"] >> enhancerParams.claheClipLimit;
        enhancerNode["clahe_tile_grid_size"] >> enhancerParams.claheTileGridSize;


        // CUDA
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
        
        // Only set values if they're not empty
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

// Function to check if application restart is recommended for mode changes
bool shouldRestart(bool currentPassthrough, bool newPassthrough) {
    return currentPassthrough != newPassthrough;
}

int main(int argc, char** argv) {
    // Set up signal handlers for graceful shutdown
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

    // Before entering your main loop, store the last modification time:
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
    trackerParams.processingWidth = 640;   // Optimal for ResNet18
    trackerParams.processingHeight = 368;  // Optimal for ResNet18
    trackerParams.confidenceThreshold = 0.3; // Lower threshold to detect more objects

    // Read the config file
    if (!readConfig(configFile, videoSource, runParams, enhancerParams, rollParams, stabParams, camParams, trackerParams)) {
        return 1; // Exit if config cannot be loaded
    }

    std::cout << "Using video source: " << videoSource << std::endl;
    std::cout << "Roll Correction: " << (runParams.rollCorrectionEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Stabilizer: " << (runParams.stabilizationEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Tracker: " << (runParams.trackerEnabled ? "Enabled" : "Disabled") << std::endl;

    vs::Stabilizer stab(stabParams);
    camParams.source = videoSource;

    // Control variables for runtime monitoring
    int emptyFrameCount = 0;
    int configCheckCounter = 0;

    std::unique_ptr<vs::CamCap> cam = std::make_unique<vs::CamCap>(camParams);
    cam->start();
    
    // Initialize tracker and TCP receiver for tracking coordinates
    std::unique_ptr<vs::DeepStreamTracker> tracker = std::make_unique<vs::DeepStreamTracker>(trackerParams);
    vs::TcpReciever tcp(5000);   // listen on port 5000
    tcp.start();
    int x = -1, y = -1;  // Tracking coordinates
    // cam->start();

    // Make sure to get the ACTUAL frame dimensions before setting up RTSP server
    double fps = cam->getFrameRate();
    if (fps < 1.0) fps = 30.0;
    std::cout << "Video framerate: " << fps << " FPS" << std::endl;
    
    // Get frame dimensions from camera properties instead of reading a frame
    int frameWidth = static_cast<int>(cam->getWidth());
    int frameHeight = static_cast<int>(cam->getHeight());
    
    // Override with config values if specified
    if (runParams.width > 0 && runParams.height > 0) {
        frameWidth = runParams.width;
        frameHeight = runParams.height;
    }
    
    std::cout << "Frame dimensions: " << frameWidth << "x" << frameHeight << std::endl;

    // Initialize variables for different modes
    std::unique_ptr<GStreamerPassthrough> passthroughStreamer;
    
    // Frame buffer management for streaming
    const int maxBufferedFrames = 2;  // Keep buffer very small for low latency
    int bufferedFrameCount = 0;

    // Check if we can use passthrough mode (no processing needed)
    bool usePassthrough = !runParams.enhancerEnabled && 
                         !runParams.rollCorrectionEnabled && 
                         !runParams.stabilizationEnabled &&
                         !runParams.trackerEnabled;
    
    if (usePassthrough && videoSource.find("rtsp://") == 0) {
        std::cout << "No processing enabled - using ultra-low latency GStreamer passthrough mode" << std::endl;
        std::cout << "Make sure MediaMTX is running on port 8554" << std::endl;
        
        // Create and start GStreamer passthrough
        passthroughStreamer = std::make_unique<GStreamerPassthrough>(
            videoSource, 
            "rtsp://localhost:8554/forwarded"
        );
        
        if (passthroughStreamer->start()) {
            std::cout << "GStreamer passthrough mode active - press Ctrl+C to stop" << std::endl;
            
            // Monitor for config changes and stop signal
            while (!stopRequested && passthroughStreamer->isActive()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                
                // Check for config changes periodically
                static int passthroughConfigCounter = 0;
                if (passthroughConfigCounter++ % 5 == 0) {  // Check every 5 seconds
                    if (stat(configFile.c_str(), &configStat) == 0) {
                        if (configStat.st_mtime != lastConfigModTime) {
                            std::cout << "\n=== Config file changed in passthrough mode ===" << std::endl;
                            
                            if (readConfig(configFile, videoSource, runParams, enhancerParams, rollParams, stabParams, camParams, trackerParams)) {
                                lastConfigModTime = configStat.st_mtime;
                                
                                bool newUsePassthrough = !runParams.enhancerEnabled && 
                                                       !runParams.rollCorrectionEnabled && 
                                                       !runParams.stabilizationEnabled &&
                                                       !runParams.trackerEnabled;
                                
                                if (!newUsePassthrough) {
                                    std::cout << "Processing enabled - exiting passthrough mode" << std::endl;
                                    std::cout << "Enhancer: " << (runParams.enhancerEnabled ? "Enabled" : "Disabled") << std::endl;
                                    std::cout << "Roll Correction: " << (runParams.rollCorrectionEnabled ? "Enabled" : "Disabled") << std::endl;
                                    std::cout << "Stabilizer: " << (runParams.stabilizationEnabled ? "Enabled" : "Disabled") << std::endl;
                                    std::cout << "Tracker: " << (runParams.trackerEnabled ? "Enabled" : "Disabled") << std::endl;
                                    
                                    // Stop passthrough
                                    passthroughStreamer->stop();
                                    passthroughStreamer.reset();
                                    
                                    // Clean shutdown to avoid conflicts
                                    std::cout << "Stopping camera for mode transition..." << std::endl;
                                    if (cam) {
                                        cam->stop();
                                        cam.reset();  // Clean shutdown
                                    }
                                    
                                    // Wait for clean shutdown
                                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                    
                                    // Exit passthrough mode
                                    usePassthrough = false;
                                    break;
                                } else {
                                    std::cout << "Still in passthrough mode - no processing enabled" << std::endl;
                                }
                            } else {
                                std::cerr << "Failed to reload configuration in passthrough mode." << std::endl;
                            }
                        }
                    }
                }
            }
            
            if (passthroughStreamer) {
                passthroughStreamer->stop();
                passthroughStreamer.reset();
            }
            
            // If we exited passthrough due to config change, continue to processing mode
            if (!usePassthrough) {
                std::cout << "Transitioning from passthrough to processing mode..." << std::endl;
                
                // Reinitialize all components for processing mode
                std::cout << "Reinitializing components for processing mode..." << std::endl;
                
                // 1. Reinitialize camera with updated parameters
                camParams.source = videoSource;
                cam = std::make_unique<vs::CamCap>(camParams);
                cam->start();
                
                // 2. Update frame parameters
                fps = cam->getFrameRate();
                if (fps < 1.0) fps = 30.0;
                frameWidth = static_cast<int>(cam->getWidth());
                frameHeight = static_cast<int>(cam->getHeight());
                
                if (runParams.width > 0 && runParams.height > 0) {
                    frameWidth = runParams.width;
                    frameHeight = runParams.height;
                }
                
                std::cout << "Processing mode - Video framerate: " << fps << " FPS" << std::endl;
                std::cout << "Processing mode - Frame dimensions: " << frameWidth << "x" << frameHeight << std::endl;
                
                // 3. Reinitialize stabilizer if needed
                if (runParams.stabilizationEnabled) {
                    stab = vs::Stabilizer(stabParams);
                }
                
                // 4. Make sure tracker is properly initialized if needed
                if (runParams.trackerEnabled) {
                    try {
                        tracker = std::make_unique<vs::DeepStreamTracker>(trackerParams);
                    } catch (const std::exception& e) {
                        std::cerr << "Failed to initialize tracker: " << e.what() << std::endl;
                    }
                }
                
                std::cout << "Component reinitialization complete!" << std::endl;
                // Don't return, continue to processing mode below
            } else {
                return 0;  // Normal exit
            }
        } else {
            std::cerr << "Failed to start GStreamer passthrough, falling back to frame processing" << std::endl;
        }
    }

    // If we reach here, either passthrough failed or processing is enabled
    std::cout << "Starting frame processing mode" << std::endl;
    
    // Setup GStreamer pipeline for more efficient streaming (like tracker_example)
    GstElement* pipeline = nullptr;
    GstAppSrc* appsrc = nullptr;
    uint64_t frameCounter = 0;
    
    // Calculate appropriate bitrate based on resolution
    int bitrate = (frameWidth * frameHeight * fps * 0.1) / 1000; // In Kbps
    if (bitrate < 800) bitrate = 800; // Minimum bitrate floor
    if (bitrate > 8000) bitrate = 8000; // Maximum bitrate ceiling
    
    auto buildStreamer = [&]() {
        gst_init(nullptr, nullptr);

        std::string pipe =
            "appsrc name=src is-live=true format=time block=false "
            "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) +
            ",height=" + std::to_string(frameHeight) +
            ",framerate=" + std::to_string(int(fps)) + "/1 ! "
            "queue max-size-buffers=2 leaky=downstream ! "  // Very small buffer for low latency
            "videoconvert ! video/x-raw,format=NV12 ! "
            "x264enc threads=4 tune=zerolatency speed-preset=veryfast "
            "bitrate=" + std::to_string(bitrate) + " ! "
            "rtspclientsink location=rtsp://localhost:8554/forwarded protocols=tcp";

        pipeline = gst_parse_launch(pipe.c_str(), nullptr);
        appsrc = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(pipeline), "src"));
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
    };

    buildStreamer();

    int delayMs = (runParams.optimizeFps) ? 1 : static_cast<int>(1000.0 / fps);

    const int windowWidth = runParams.width;
    const int windowHeight = runParams.height;

    // Only create windows if not optimizing for FPS
    if (!runParams.optimizeFps) {
        cv::namedWindow("Raw", cv::WINDOW_NORMAL);
        cv::resizeWindow("Raw", windowWidth, windowHeight);

        cv::namedWindow("Final", cv::WINDOW_NORMAL);
        cv::resizeWindow("Final", windowWidth, windowHeight);
    }

    while (!stopRequested) {
        // Check if the camera is still healthy - less aggressive checking
        if (!cam->isHealthy()) {
            std::cout << "Camera/stream is not healthy, attempting to restart..." << std::endl;
            cam->stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // Wait 2 seconds before restart
            
            // Try to restart the camera
            try {
                camParams.source = videoSource;
                cam = std::make_unique<vs::CamCap>(camParams);
                cam->start();
                std::cout << "Camera restarted successfully!" << std::endl;
                
                // Reset counters after successful restart
                emptyFrameCount = 0;
                configCheckCounter = 0;
                
            } catch (const std::exception& e) {
                std::cerr << "Failed to restart camera: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10000)); // Wait 10 seconds before retry
                continue;
            }
        }
        
        // Check if the config file has been modified (do this less frequently)
        static int configCheckCounter = 0;
        if (configCheckCounter++ % 30 == 0) {  // Check every 30 frames (~1 second at 30fps)
            if (stat(configFile.c_str(), &configStat) == 0) {
                if (configStat.st_mtime != lastConfigModTime) {
                    std::cout << "\n=== Configuration file updated, reloading parameters... ===" << std::endl;
                    
                    // Store old values for comparison
                    std::string oldVideoSource = videoSource;
                    bool oldEnhancer = runParams.enhancerEnabled;
                    bool oldRollCorrection = runParams.rollCorrectionEnabled;
                    bool oldStabilizer = runParams.stabilizationEnabled;
                    int oldWidth = frameWidth;
                    int oldHeight = frameHeight;
                    
                    if (readConfig(configFile, videoSource, runParams, enhancerParams, rollParams, stabParams, camParams, trackerParams)) {
                        lastConfigModTime = configStat.st_mtime;
                        
                        std::cout << "Roll Correction: " << (runParams.rollCorrectionEnabled ? "Enabled" : "Disabled") << std::endl;
                        std::cout << "Stabilizer: " << (runParams.stabilizationEnabled ? "Enabled" : "Disabled") << std::endl;
                        std::cout << "Enhancer: " << (runParams.enhancerEnabled ? "Enabled" : "Disabled") << std::endl;
                        
                        // 1. Update Stabilizer if parameters changed
                        if (runParams.stabilizationEnabled || oldStabilizer) {
                            std::cout << "Reinitializing stabilizer..." << std::endl;
                            stab = vs::Stabilizer(stabParams);
                        }
                        
                        // 2. Update camera if source changed
                        camParams.source = videoSource;
                        if (videoSource != oldVideoSource) {
                            std::cout << "Video source changed from " << oldVideoSource << " to " << videoSource << std::endl;
                            
                            // Stop the current camera
                            cam->stop();
                            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Give time to stop
                            
                            // Reinitialize camera with new source
                            cam = std::make_unique<vs::CamCap>(camParams);
                            cam->start();

                            
                            // Get fresh frame rate after camera restart
                            fps = cam->getFrameRate();
                            if (fps < 1.0) fps = 30.0;
                            std::cout << "Updated video framerate: " << fps << " FPS" << std::endl;
                            
                            // Update delay time for main loop
                            delayMs = (runParams.optimizeFps) ? 1 : static_cast<int>(1000.0 / fps);
                        }
                        
                        // 3. Update frame dimensions if changed
                        int newFrameWidth = static_cast<int>(cam->getWidth());
                        int newFrameHeight = static_cast<int>(cam->getHeight());
                        
                        if (runParams.width > 0 && runParams.height > 0) {
                            newFrameWidth = runParams.width;
                            newFrameHeight = runParams.height;
                        }
                        
                        if (newFrameWidth != frameWidth || newFrameHeight != frameHeight) {
                            std::cout << "Frame dimensions changed from " << frameWidth << "x" << frameHeight 
                                     << " to " << newFrameWidth << "x" << newFrameHeight << std::endl;
                            frameWidth = newFrameWidth;
                            frameHeight = newFrameHeight;
                            
                            // Restart GStreamer pipeline with new dimensions
                            if (appsrc && pipeline) {
                                std::cout << "Restarting GStreamer pipeline with new frame dimensions..." << std::endl;
                                gst_element_set_state(pipeline, GST_STATE_NULL);
                                gst_object_unref(pipeline);
                                
                                // Rebuild GStreamer pipeline with new dimensions
                                buildStreamer();
                                
                                if (!appsrc) {
                                    std::cerr << "Failed to restart GStreamer pipeline with new dimensions" << std::endl;
                                }
                            }
                        }
                        
                        // 4. Update window sizes if dimensions changed and windows are enabled
                        if (!runParams.optimizeFps && frameWidth > 0 && frameHeight > 0) {
                            cv::resizeWindow("Raw", frameWidth, frameHeight);
                            cv::resizeWindow("Final", frameWidth, frameHeight);
                        }
                        
                        // 5. Check if we need to switch between passthrough and processing mode
                        bool newUsePassthrough = !runParams.enhancerEnabled && 
                                               !runParams.rollCorrectionEnabled && 
                                               !runParams.stabilizationEnabled;
                        
                        if (newUsePassthrough && !usePassthrough && videoSource.find("rtsp://") == 0) {
                            std::cout << "Switching to passthrough mode for optimal performance" << std::endl;
                            
                            // Stop all processing components cleanly
                            std::cout << "Stopping all processing components..." << std::endl;
                            
                            // Stop tracker first
                            std::cout << "Stopping tracker..." << std::endl;
                            tracker.reset();
                            std::cout << "Tracker stopped." << std::endl;
                            
                            // Stop camera
                            std::cout << "Stopping camera..." << std::endl;
                            cam->stop();
                            cam.reset();
                            std::cout << "Camera stopped." << std::endl;
                            
                            // Stop GStreamer pipeline
                            if (appsrc && pipeline) {
                                std::cout << "Stopping GStreamer pipeline..." << std::endl;
                                gst_element_set_state(pipeline, GST_STATE_NULL);
                                gst_object_unref(pipeline);
                                pipeline = nullptr;
                                appsrc = nullptr;
                                std::cout << "GStreamer pipeline stopped." << std::endl;
                            }
                            
                            std::cout << "All processing components stopped." << std::endl;
                            
                            // Start GStreamer passthrough mode
                            passthroughStreamer = std::make_unique<GStreamerPassthrough>(
                                videoSource, 
                                "rtsp://localhost:8554/forwarded"
                            );
                            
                            if (passthroughStreamer->start()) {
                                std::cout << "GStreamer passthrough activated successfully!" << std::endl;
                                usePassthrough = true;
                                
                                // Exit the processing loop to run passthrough
                                std::cout << "Exiting processing loop for passthrough mode..." << std::endl;
                                break;
                            } else {
                                std::cerr << "Failed to start GStreamer passthrough, restarting processing components..." << std::endl;
                                
                                // Restart processing components
                                camParams.source = videoSource;
                                cam = std::make_unique<vs::CamCap>(camParams);
                                cam->start();
                                
                                // Reinitialize tracker if needed
                                if (runParams.trackerEnabled) {
                                    tracker = std::make_unique<vs::DeepStreamTracker>(trackerParams);
                                }
                                
                                // Restart GStreamer pipeline
                                buildStreamer();
                            }
                        } else if (!newUsePassthrough && usePassthrough) {
                            std::cout << "Note: Processing mode enabled but currently in passthrough - restart recommended for optimal performance" << std::endl;
                        }
                        
                        std::cout << "=== Configuration reloaded successfully ===" << std::endl;
                    } else {
                        std::cerr << "Failed to reload configuration." << std::endl;
                    }
                }
            }
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        cv::Mat frame = cam->read();
        
        static int emptyFrameCount = 0;
        if (frame.empty()) {
            // Count consecutive empty frames to detect stream issues
            emptyFrameCount++;
            
            if (emptyFrameCount > 30) { // More lenient - allow 30 consecutive empty frames
                std::cout << "Too many empty frames (" << emptyFrameCount << "), camera may have disconnected" << std::endl;
                emptyFrameCount = 0; // Reset counter
                continue; // This will trigger the health check at the beginning of the loop
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Longer pause to reduce CPU usage
            continue;
        } else {
            emptyFrameCount = 0; // Reset on successful frame read
        }

        // Skip expensive display operations if optimizing for FPS
        static int frameSkipCounter = 0;
        
        // Only show raw frame very occasionally to reduce processing overhead
        if (!runParams.optimizeFps && frameSkipCounter % 30 == 0) { // Changed from 10 to 30
            cv::Mat displayFrame;
            if (windowWidth > 0 && windowHeight > 0) {
                cv::resize(frame, displayFrame, cv::Size(windowWidth, windowHeight));
                cv::imshow("Raw", displayFrame);
            } else {
                cv::imshow("Raw", frame);
            }
        }

        // Apply processing only if enabled - avoid unnecessary copying
        cv::Mat* framePtr = &frame;  // Use pointer to avoid copying
        cv::Mat tempFrame1, tempFrame2, tempFrame3;  // Reuse these instead of creating new ones
        
        if (runParams.enhancerEnabled) {
            tempFrame1 = vs::Enhancer::enhanceImage(*framePtr, enhancerParams);
            framePtr = &tempFrame1;
        }

        // Apply Roll Correction
        if (runParams.rollCorrectionEnabled) {
            tempFrame2 = vs::RollCorrection::autoCorrectRoll(*framePtr, rollParams);
            framePtr = &tempFrame2;
        }

        // Apply Stabilization
        if (runParams.stabilizationEnabled) {
            tempFrame3 = stab.stabilize(*framePtr);
            framePtr = &tempFrame3;
        }
        
        // Apply Tracking
        if (runParams.trackerEnabled && tracker) {
            // Process frame through tracker
            auto detections = tracker->processFrame(*framePtr);
            
            // Check for new tracking coordinates from TCP
            if (tcp.tryGetLatest(x, y)) {
                std::cout << "Received tracking coordinates: (" << x << "," << y << ") with " 
                          << detections.size() << " detections available" << std::endl;
                
                // Draw detections with the selected coordinates
                cv::Mat trackedFrame = tracker->drawDetections(*framePtr, detections, x, y);
                if (framePtr == &tempFrame3) {
                    tempFrame3 = trackedFrame;  // Update the existing frame
                } else {
                    tempFrame3 = trackedFrame;  // Use tempFrame3 for tracked output
                    framePtr = &tempFrame3;
                }
            } else {
                // No new coordinates, use previous selection
                cv::Mat trackedFrame = tracker->drawDetections(*framePtr, detections, -1, -1);
                if (framePtr == &tempFrame3) {
                    tempFrame3 = trackedFrame;  // Update the existing frame
                } else {
                    tempFrame3 = trackedFrame;  // Use tempFrame3 for tracked output
                    framePtr = &tempFrame3;
                }
            }
        }
        
        cv::Mat& processedFrame = *framePtr;  // Reference to final processed frame

        // Only display final output very occasionally to reduce overhead
        if (!runParams.optimizeFps && frameSkipCounter % 30 == 0 && !processedFrame.empty()) { // Changed from 10 to 30
            cv::Mat displayFrame;
            if (windowWidth > 0 && windowHeight > 0) {
                cv::resize(processedFrame, displayFrame, cv::Size(windowWidth, windowHeight));
                cv::imshow("Final", displayFrame);
            } else {
                cv::imshow("Final", processedFrame);
            }
        }
        
        // Send frame to MediaMTX via GStreamer (more efficient than FFmpeg pipe)
        if (!processedFrame.empty() && appsrc) {
        // Improved timing control to reduce glitches
        static auto lastSendTime = std::chrono::high_resolution_clock::now();
        static int frameDropCount = 0;
        auto currentTime = std::chrono::high_resolution_clock::now();
        double timeSinceLastSend = std::chrono::duration<double, std::milli>(currentTime - lastSendTime).count();
        
        // More precise frame timing
        double targetInterval = 1000.0 / fps;
        
        // Only send frame if we're not getting too far behind
        if (timeSinceLastSend >= targetInterval * 0.9) {  // Tighter timing control
            // Ensure frame is properly formatted and sized
            cv::Mat outputFrame;
            
            // Always ensure the frame is continuous and properly formatted
            if (processedFrame.cols != frameWidth || processedFrame.rows != frameHeight) {
                cv::resize(processedFrame, outputFrame, cv::Size(frameWidth, frameHeight), 0, 0, cv::INTER_LANCZOS4);
            } else {
                // Ensure frame is continuous in memory
                if (!processedFrame.isContinuous()) {
                    processedFrame.copyTo(outputFrame);
                } else {
                    outputFrame = processedFrame.clone(); // Create a copy to avoid memory issues
                }
            }
            
            // Ensure BGR format (GStreamer expects this)
            if (outputFrame.channels() != 3) {
                cv::cvtColor(outputFrame, outputFrame, cv::COLOR_GRAY2BGR);
            }
            
            // Create GStreamer buffer with proper size
            size_t bufferSize = outputFrame.total() * outputFrame.elemSize();
            GstBuffer* buf = gst_buffer_new_allocate(nullptr, bufferSize, nullptr);
            
            GstMapInfo map;
            if (gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
                // Copy frame data
                std::memcpy(map.data, outputFrame.data, bufferSize);
                gst_buffer_unmap(buf, &map);
                
                // Set proper timestamp
                GST_BUFFER_PTS(buf) = gst_util_uint64_scale(frameCounter++, GST_SECOND, fps);
                GST_BUFFER_DURATION(buf) = gst_util_uint64_scale(1, GST_SECOND, fps);
                
                // Push buffer to pipeline
                GstFlowReturn ret = gst_app_src_push_buffer(appsrc, buf);
                if (ret != GST_FLOW_OK) {
                    std::cerr << "Failed to push buffer to GStreamer pipeline: " << ret << std::endl;
                    frameDropCount++;
                    if (frameDropCount > 10) {
                        std::cout << "Too many failed pushes, restarting pipeline..." << std::endl;
                        // Restart pipeline logic here if needed
                        frameDropCount = 0;
                    }
                } else {
                    frameDropCount = 0; // Reset on success
                }
                
                lastSendTime = currentTime;
            } else {
                std::cerr << "Failed to map GStreamer buffer" << std::endl;
                gst_buffer_unref(buf);
            }
        }
    }
        

        // Measure Processing Time and adapt performance
        auto endTime = std::chrono::high_resolution_clock::now();
        double frameTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        frameSkipCounter++;
        
        // Print performance stats much less frequently to reduce console overhead
        if (frameSkipCounter % 300 == 0) {  // Every 300 frames (~10 seconds at 30fps)
            double currentFps = 1000.0 / frameTime;
            std::cout << "Processing Time: " << frameTime << " ms | FPS: " << currentFps << std::endl;
        }

        // Adaptive delay based on processing time to maintain target FPS
        if (!runParams.optimizeFps) {
            double targetFrameTime = 1000.0 / fps;
            if (frameTime < targetFrameTime) {
                double sleepTime = targetFrameTime - frameTime;
                if (sleepTime > 1.0) {  // Only sleep if significant time available
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(sleepTime * 1000)));
                }
            }
        }

        // Minimal delay for key detection - only if not optimizing
        if (!runParams.optimizeFps && frameSkipCounter % 10 == 0) {  // Check keys less frequently
            if (cv::waitKey(1) == 27) {
                std::cout << "ESC key pressed, stopping..." << std::endl;
                break;
            }
        }
    }
    
    // If we switched to passthrough mode, run the passthrough loop
    if (usePassthrough && passthroughStreamer) {
        std::cout << "Entering passthrough mode loop..." << std::endl;
        
        // Simple monitoring loop for passthrough mode
        while (!stopRequested && passthroughStreamer->isActive()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            // Check for config changes periodically
            static int passthroughConfigCounter = 0;
            if (passthroughConfigCounter++ % 10 == 0) {  // Check every ~10 seconds
                if (stat(configFile.c_str(), &configStat) == 0) {
                    if (configStat.st_mtime != lastConfigModTime) {
                        std::cout << "\n=== Config changed in passthrough mode ===" << std::endl;
                        
                        if (readConfig(configFile, videoSource, runParams, enhancerParams, rollParams, stabParams, camParams, trackerParams)) {
                            lastConfigModTime = configStat.st_mtime;
                            
                            bool newUsePassthrough = !runParams.enhancerEnabled && 
                                                   !runParams.rollCorrectionEnabled && 
                                                   !runParams.stabilizationEnabled &&
                                                   !runParams.trackerEnabled;
                            
                            if (!newUsePassthrough) {
                                std::cout << "Processing enabled - exiting passthrough mode" << std::endl;
                                passthroughStreamer->stop();
                                passthroughStreamer.reset();
                                break;
                            } else {
                                std::cout << "Still in passthrough mode" << std::endl;
                            }
                        }
                    }
                }
            }
        }
        
        std::cout << "Passthrough mode loop ended" << std::endl;
    }
    
    std::cout << "Cleaning up resources..." << std::endl;
    
    // Stop camera capture
    if (cam) {
        cam->stop();
    }
    
    // Cleanup GStreamer pipeline
    if (appsrc && pipeline) {
        std::cout << "Closing GStreamer pipeline..." << std::endl;
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        pipeline = nullptr;
        appsrc = nullptr;
    }
    
    // Stop TCP receiver
    tcp.stop();
    
    // Cleanup GStreamer passthrough
    if (passthroughStreamer) {
        std::cout << "Stopping GStreamer passthrough..." << std::endl;
        passthroughStreamer->stop();
        passthroughStreamer.reset();
        std::cout << "GStreamer passthrough stopped." << std::endl;
    }
    
    cv::destroyAllWindows();
    std::cout << "Cleanup complete." << std::endl;
    return 0;
}
