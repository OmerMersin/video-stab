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
    
    // GStreamer elements
    GstElement* sourcePipeline;
    GstElement* outputPipeline;
    GstElement* passthroughPipeline;
    GstElement* processingPipeline;
    
    // AppSrc elements for feeding processed data
    GstElement* passthroughAppSrc;
    GstElement* processingAppSrc;
    
    // Inter-pipeline elements
    GstElement* sourceSink;
    GstElement* passthroughSink;
    GstElement* processingSink;
    GstElement* outputSrc;
    
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
                if (GST_MESSAGE_SRC(msg) == GST_OBJECT(manager->sourcePipeline) ||
                    GST_MESSAGE_SRC(msg) == GST_OBJECT(manager->outputPipeline) ||
                    GST_MESSAGE_SRC(msg) == GST_OBJECT(manager->passthroughPipeline) ||
                    GST_MESSAGE_SRC(msg) == GST_OBJECT(manager->processingPipeline)) {
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
    
    bool checkInterpipePlugin() {
        GstRegistry* registry = gst_registry_get();
        GstPlugin* plugin = gst_registry_find_plugin(registry, "interpipe");
        
        if (!plugin) {
            std::cerr << "Error: interpipe plugin not found!" << std::endl;
            std::cerr << "Please install gstreamer1.0-interpipe package" << std::endl;
            return false;
        }
        
        gst_object_unref(plugin);
        return true;
    }
    
public:
    GStreamerPipelineManager(const std::string& source = "rtsp://192.168.144.119:554",
                           const std::string& output = "rtsp://192.168.144.150:8554/forwarded", 
                           int bitrate = 4000000) 
        : sourceAddress(source), outputAddress(output), bitrate(bitrate), pipelinesInitialized(false),
          sourcePipeline(nullptr), outputPipeline(nullptr), passthroughPipeline(nullptr), processingPipeline(nullptr),
          passthroughAppSrc(nullptr), processingAppSrc(nullptr),
          sourceSink(nullptr), passthroughSink(nullptr), processingSink(nullptr), outputSrc(nullptr),
          isCurrentlyPassthrough(true) {
        
        // Initialize GStreamer
        if (!gst_is_initialized()) {
            gst_init(nullptr, nullptr);
        }
    }
    
   bool initialize(int width, int height, double framerate) {
        frameWidth = width;
        frameHeight = height;
        fps = framerate;
        
        // Check if interpipe plugin is available
        if (!checkInterpipePlugin()) {
            return false;
        }

        // Create source pipeline - receives RTSP H.265 stream
        std::string sourcePipelineStr = 
            "rtspsrc name=src location=" + sourceAddress + " latency=1000 ! "
            "rtph265depay ! h265parse ! "
            "interpipesink name=to_output sync=false async=false format-time=true";
        
        sourcePipeline = gst_parse_launch(sourcePipelineStr.c_str(), nullptr);
        if (!sourcePipeline) {
            std::cerr << "Failed to create source pipeline" << std::endl;
            return false;
        }

        // Create output pipeline - forwards to output RTSP
        std::string outputPipelineStr = 
            "interpipesrc name=from_input listen-to=to_output is-live=true do-timestamp=true format=time ! "
            "rtspclientsink location=" + outputAddress;
        
        outputPipeline = gst_parse_launch(outputPipelineStr.c_str(), nullptr);
        if (!outputPipeline) {
            std::cerr << "Failed to create output pipeline" << std::endl;
            return false;
        }

        // Create passthrough pipeline for processed frames
        std::string passthroughPipelineStr = 
            "appsrc name=passthrough_src is-live=true format=time block=false max-latency=0 "
            "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) + 
            ",height=" + std::to_string(frameHeight) + 
            ",framerate=" + std::to_string(static_cast<int>(fps)) + "/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! video/x-raw,format=NV12 ! "
            "x264enc threads=4 tune=zerolatency speed-preset=ultrafast "
            "bitrate=" + std::to_string(bitrate / 1000) + " key-int-max=15 intra-refresh=true "
            "b-adapt=0 bframes=0 ref=1 me=dia subme=0 trellis=0 weightp=0 "
            "rc-lookahead=0 sync-lookahead=0 sliced-threads=true "
            "aud=false annexb=false ! "
            "h264parse ! "
            "interpipesink name=passthrough_sink sync=false async=false format-time=true";
        
        passthroughPipeline = gst_parse_launch(passthroughPipelineStr.c_str(), nullptr);
        if (!passthroughPipeline) {
            std::cerr << "Failed to create passthrough pipeline" << std::endl;
            return false;
        }

        // Create processing pipeline for processed frames
        std::string processingPipelineStr = 
            "appsrc name=processing_src is-live=true format=time block=false max-latency=0 "
            "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) + 
            ",height=" + std::to_string(frameHeight) + 
            ",framerate=" + std::to_string(static_cast<int>(fps)) + "/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! video/x-raw,format=NV12 ! "
            "x264enc threads=4 tune=zerolatency speed-preset=ultrafast "
            "bitrate=" + std::to_string(bitrate / 1000) + " key-int-max=15 intra-refresh=true "
            "b-adapt=0 bframes=0 ref=1 me=dia subme=0 trellis=0 weightp=0 "
            "rc-lookahead=0 sync-lookahead=0 sliced-threads=true "
            "aud=false annexb=false ! "
            "h264parse ! "
            "interpipesink name=processing_sink sync=false async=false format-time=true";
        
        processingPipeline = gst_parse_launch(processingPipelineStr.c_str(), nullptr);
        if (!processingPipeline) {
            std::cerr << "Failed to create processing pipeline" << std::endl;
            return false;
        }

        // Get references to important elements
        passthroughAppSrc = gst_bin_get_by_name(GST_BIN(passthroughPipeline), "passthrough_src");
        processingAppSrc = gst_bin_get_by_name(GST_BIN(processingPipeline), "processing_src");
        sourceSink = gst_bin_get_by_name(GST_BIN(sourcePipeline), "to_output");
        passthroughSink = gst_bin_get_by_name(GST_BIN(passthroughPipeline), "passthrough_sink");
        processingSink = gst_bin_get_by_name(GST_BIN(processingPipeline), "processing_sink");
        outputSrc = gst_bin_get_by_name(GST_BIN(outputPipeline), "from_input");

        if (!passthroughAppSrc || !processingAppSrc || !sourceSink || !passthroughSink || !processingSink || !outputSrc) {
            std::cerr << "Failed to get pipeline elements" << std::endl;
            return false;
        }

        // Configure AppSrc elements with proper timing
        g_object_set(passthroughAppSrc,
            "is-live", TRUE,
            "block", FALSE,
            "format", GST_FORMAT_TIME,
            "max-latency", G_GINT64_CONSTANT(0),
            "do-timestamp", TRUE,
            "min-latency", G_GINT64_CONSTANT(0),
            NULL);
            
        g_object_set(processingAppSrc,
            "is-live", TRUE,
            "block", FALSE,
            "format", GST_FORMAT_TIME,
            "max-latency", G_GINT64_CONSTANT(0),
            "do-timestamp", TRUE,
            "min-latency", G_GINT64_CONSTANT(0),
            NULL);

        // Configure interpipe elements for consistent timing
        g_object_set(sourceSink,
            "sync", FALSE,
            "async", FALSE,
            "format-time", TRUE,
            NULL);
            
        g_object_set(passthroughSink,
            "sync", FALSE,
            "async", FALSE,
            "format-time", TRUE,
            NULL);
            
        g_object_set(processingSink,
            "sync", FALSE,
            "async", FALSE,
            "format-time", TRUE,
            NULL);
            
        g_object_set(outputSrc,
            "is-live", TRUE,
            "do-timestamp", TRUE,
            "format", GST_FORMAT_TIME,
            NULL);
        
        // Set up bus message handling
        GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(sourcePipeline));
        gst_bus_add_watch(bus, busCallback, this);
        gst_object_unref(bus);
        
        bus = gst_pipeline_get_bus(GST_PIPELINE(outputPipeline));
        gst_bus_add_watch(bus, busCallback, this);
        gst_object_unref(bus);
        
        bus = gst_pipeline_get_bus(GST_PIPELINE(passthroughPipeline));
        gst_bus_add_watch(bus, busCallback, this);
        gst_object_unref(bus);
        
        bus = gst_pipeline_get_bus(GST_PIPELINE(processingPipeline));
        gst_bus_add_watch(bus, busCallback, this);
        gst_object_unref(bus);
        
        // Start all pipelines
        if (gst_element_set_state(sourcePipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start source pipeline" << std::endl;
            return false;
        }
        
        if (gst_element_set_state(outputPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start output pipeline" << std::endl;
            return false;
        }
        
        if (gst_element_set_state(passthroughPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start passthrough pipeline" << std::endl;
            return false;
        }
        
        if (gst_element_set_state(processingPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start processing pipeline" << std::endl;
            return false;
        }
        
        // Wait for state changes
        gst_element_get_state(sourcePipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
        gst_element_get_state(outputPipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
        gst_element_get_state(passthroughPipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
        gst_element_get_state(processingPipeline, nullptr, nullptr, GST_CLOCK_TIME_NONE);
        
        pipelinesInitialized = true;
        std::cout << "GStreamer Pipeline Manager initialized successfully" << std::endl;
        std::cout << "Source: " << sourceAddress << std::endl;
        std::cout << "Output: " << outputAddress << std::endl;
        return true;
    }
    
    bool switchToPassthrough() {
        if (!pipelinesInitialized) {
            std::cerr << "Pipelines not initialized" << std::endl;
            return false;
        }
        
        if (isCurrentlyPassthrough) {
            std::cout << "Already in passthrough mode - using direct RTSP stream" << std::endl;
            return true;
        }
        
        // Switch the output interpipesrc to listen to the original stream
        g_object_set(outputSrc, "listen-to", "to_output", nullptr);
        
        isCurrentlyPassthrough = true;
        std::cout << "Switched to passthrough mode - direct RTSP forwarding" << std::endl;
        return true;
    }
    
    bool switchToProcessing() {
        if (!pipelinesInitialized) {
            std::cerr << "Pipelines not initialized" << std::endl;
            return false;
        }
        
        if (!isCurrentlyPassthrough) {
            std::cout << "Already in processing mode" << std::endl;
            return true;
        }
        
        // Switch the output interpipesrc to listen to processing_sink
        g_object_set(outputSrc, "listen-to", "processing_sink", nullptr);
        
        isCurrentlyPassthrough = false;
        std::cout << "Switched to processing mode - using processed frames" << std::endl;
        return true;
    }
    
        bool pushFrame(const cv::Mat& frame) {
        if (!pipelinesInitialized) {
            return false;
        }
        
        // Only push frames if we're not in passthrough mode
        if (isCurrentlyPassthrough) {
            return true; // In passthrough mode, frames go directly through RTSP
        }
        
        // Ensure the frame has the correct dimensions
        cv::Mat outputFrame;
        if (frame.cols != frameWidth || frame.rows != frameHeight) {
            cv::resize(frame, outputFrame, cv::Size(frameWidth, frameHeight), 0, 0, cv::INTER_LINEAR);
        } else {
            outputFrame = frame.clone();
        }
        
        // Ensure BGR format
        if (outputFrame.channels() != 3) {
            cv::cvtColor(outputFrame, outputFrame, cv::COLOR_GRAY2BGR);
        }
        
        // Create GStreamer buffer from OpenCV Mat
        size_t bufferSize = outputFrame.total() * outputFrame.elemSize();
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bufferSize, nullptr);
        
        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            gst_buffer_unref(buffer);
            return false;
        }
        
        memcpy(map.data, outputFrame.data, bufferSize);
        gst_buffer_unmap(buffer, &map);
        
        // Set proper buffer timing with consistent format
        static uint64_t frameCounter = 0;
        static GstClockTime baseTime = GST_CLOCK_TIME_NONE;
        
        if (baseTime == GST_CLOCK_TIME_NONE) {
            baseTime = gst_util_get_timestamp();
        }
        
        GstClockTime timestamp = gst_util_uint64_scale(frameCounter, GST_SECOND, static_cast<uint64_t>(fps));
        GstClockTime duration = gst_util_uint64_scale(1, GST_SECOND, static_cast<uint64_t>(fps));
        
        GST_BUFFER_PTS(buffer) = timestamp;
        GST_BUFFER_DTS(buffer) = timestamp;
        GST_BUFFER_DURATION(buffer) = duration;
        
        // Mark buffer as live
        GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_LIVE);
        
        frameCounter++;
        
        // Push to processing pipeline
        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(processingAppSrc), buffer);
        
        if (ret != GST_FLOW_OK) {
            std::cerr << "Failed to push frame to processing pipeline: " << gst_flow_get_name(ret) << std::endl;
            return false;
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
        std::cout << "  - Bitrate: " << bitrate << " bps" << std::endl;
        std::cout << "  - Current Mode: " << (isCurrentlyPassthrough ? "Passthrough (Direct RTSP)" : "Processing (Encoded)") << std::endl;
        std::cout << "  - Pipelines Initialized: Yes" << std::endl;
    }
    
    void cleanup() {
        if (sourcePipeline) {
            gst_element_set_state(sourcePipeline, GST_STATE_NULL);
            gst_object_unref(sourcePipeline);
            sourcePipeline = nullptr;
        }
        
        if (outputPipeline) {
            gst_element_set_state(outputPipeline, GST_STATE_NULL);
            gst_object_unref(outputPipeline);
            outputPipeline = nullptr;
        }
        
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
        
        if (passthroughAppSrc) {
            gst_object_unref(passthroughAppSrc);
            passthroughAppSrc = nullptr;
        }
        
        if (processingAppSrc) {
            gst_object_unref(processingAppSrc);
            processingAppSrc = nullptr;
        }
        
        if (sourceSink) {
            gst_object_unref(sourceSink);
            sourceSink = nullptr;
        }
        
        if (passthroughSink) {
            gst_object_unref(passthroughSink);
            passthroughSink = nullptr;
        }
        
        if (processingSink) {
            gst_object_unref(processingSink);
            processingSink = nullptr;
        }
        
        if (outputSrc) {
            gst_object_unref(outputSrc);
            outputSrc = nullptr;
        }
        
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
    int bitrate = std::max(800, std::min(8000, static_cast<int>(frameWidth * frameHeight * fps * 0.1 / 1000)));
    
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
            // TODO: Implement frame capture from RTSP stream for processing
            // For now, we'll just maintain the pipeline switching capability
            std::cout << "Processing mode active - pipeline ready for processed frames" << std::endl;
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
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30fps
    }
    
    std::cout << "Shutting down..." << std::endl;
    
    // Stop TCP receiver
    tcp.stop();
    
    // Cleanup pipeline manager
    pipelineManager.cleanup();
    
    cv::destroyAllWindows();
    std::cout << "Cleanup complete." << std::endl;
    return 0;
}
