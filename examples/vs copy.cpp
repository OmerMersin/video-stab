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
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <future>  // For std::async


// Global variable for signal handling
volatile sig_atomic_t stopRequested = 0;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down gracefully..." << std::endl;
    stopRequested = 1;
}


class GStreamerPipelineManager {
private:
    std::string sourceAddress;
    std::string outputAddress;
    int frameWidth, frameHeight;
    double fps;
    int bitrate;
    bool pipelinesInitialized;
    
    // Interpipe-based seamless switching architecture
    GstElement* sourcePipeline;          // RTSP source to interpipesink
    GstElement* passthroughPipeline;     // interpipesrc -> direct output
    GstElement* processingPipeline;      // interpipesrc -> appsink (for receiving frames)
    GstElement* processedOutputPipeline; // appsrc -> encode -> interpipesink (for processed frames)
    GstElement* outputPipeline;          // Final output pipeline
    
    // Processing components
    GstElement* processingAppsink;    // Receives frames for processing
    GstElement* processingAppsrc;     // Sends processed frames
    GstElement* switchElement;        // Input selector for seamless switching
    
    bool streamInitialized;
    uint64_t frameCounter;
    bool isCurrentlyPassthrough;
    
    // Threading for frame processing
    std::thread processingThread;
    std::atomic<bool> processingActive;
    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    
    // GStreamer callback for appsink (receives frames for processing)
    static GstFlowReturn newSampleCallback(GstElement* appsink, gpointer data) {
        GStreamerPipelineManager* manager = static_cast<GStreamerPipelineManager*>(data);
        
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
        if (!sample) {
            return GST_FLOW_ERROR;
        }
        
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        
        // Convert GStreamer buffer to OpenCV Mat
        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            // Get frame info from caps
            GstStructure* structure = gst_caps_get_structure(caps, 0);
            int width, height;
            gst_structure_get_int(structure, "width", &width);
            gst_structure_get_int(structure, "height", &height);
            
            // Create OpenCV Mat from buffer data
            cv::Mat frame(height, width, CV_8UC3, map.data);
            cv::Mat frameCopy = frame.clone(); // Make a copy for thread safety
            
            // Add frame to processing queue
            {
                std::lock_guard<std::mutex> lock(manager->queueMutex);
                if (manager->frameQueue.size() < 5) { // Limit queue size
                    manager->frameQueue.push(frameCopy);
                    manager->queueCondition.notify_one();
                } else {
                    // Drop oldest frame if queue is full
                    manager->frameQueue.pop();
                    manager->frameQueue.push(frameCopy);
                    manager->queueCondition.notify_one();
                }
            }
            
            gst_buffer_unmap(buffer, &map);
        }
        
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    
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
                GstState old_state, new_state, pending_state;
                gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
                // Only log pipeline state changes, not element state changes
                if (GST_IS_PIPELINE(GST_MESSAGE_SRC(msg))) {
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

    // Frame processing thread method
    void frameProcessingThread() {
        std::cout << "Frame processing thread started" << std::endl;
        
        while (processingActive.load()) {
            cv::Mat frame;
            
            // Wait for frame in queue
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCondition.wait(lock, [this] { 
                    return !frameQueue.empty() || !processingActive.load(); 
                });
                
                if (!processingActive.load()) break;
                
                if (!frameQueue.empty()) {
                    frame = frameQueue.front();
                    frameQueue.pop();
                }
            }
            
            if (frame.empty()) continue;
            
            // Process frame using the provided processor function
            cv::Mat processedFrame = frame;
            if (frameProcessor && !isCurrentlyPassthrough) {
                try {
                    processedFrame = frameProcessor(frame);
                } catch (const std::exception& e) {
                    std::cerr << "Frame processing error: " << e.what() << std::endl;
                    processedFrame = frame; // Fallback to original frame
                }
            }
            
            frameCounter++;
            
            // Push processed frame back to appsrc
            if (processingAppsrc && !isCurrentlyPassthrough && !processedFrame.empty()) {
                pushProcessedFrame(processedFrame);
            }
        }
        
        std::cout << "Frame processing thread stopped" << std::endl;
    }
    
    bool pushProcessedFrame(const cv::Mat& frame) {
        if (!processingAppsrc || frame.empty()) {
            return false;
        }
        
        // Create GStreamer buffer
        size_t bufferSize = frame.total() * frame.elemSize();
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
        
        memcpy(map.data, frame.data, bufferSize);
        gst_buffer_unmap(buffer, &map);
        
        // Set timestamp
        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frameCounter, GST_SECOND, fps);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, fps);
        
        // Push buffer
        GstFlowReturn flowRet = gst_app_src_push_buffer(GST_APP_SRC(processingAppsrc), buffer);
        
        if (flowRet != GST_FLOW_OK) {
            std::cerr << "Failed to push processed frame: " << gst_flow_get_name(flowRet) << std::endl;
            return false;
        }
        
        return true;
    }
    
public:
    GStreamerPipelineManager(const std::string& source = "rtsp://192.168.144.119:554",
                           const std::string& output = "rtsp://192.168.144.150:8554/forwarded", 
                           int bitrate = 4000000) 
        : sourceAddress(source), outputAddress(output), bitrate(bitrate), pipelinesInitialized(false),
          sourcePipeline(nullptr), passthroughPipeline(nullptr), processingPipeline(nullptr),
          processedOutputPipeline(nullptr), outputPipeline(nullptr), processingAppsink(nullptr), 
          processingAppsrc(nullptr), switchElement(nullptr), streamInitialized(false), frameCounter(0), 
          isCurrentlyPassthrough(true), processingActive(false) {
        
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
        
        std::cout << "Initializing interpipe-based seamless switching system..." << std::endl;
        
        // Calculate bitrate
        int calculatedBitrate = std::max(800, std::min(8000, static_cast<int>(frameWidth * frameHeight * fps * 0.1)));
        bitrate = calculatedBitrate;
        
        // 1. SOURCE PIPELINE: RTSP source -> interpipesink
        std::string sourcePipelineStr = 
            "rtspsrc location=" + sourceAddress + " latency=0 protocols=tcp ! "
            "rtph264depay ! h264parse ! "
            "interpipesink name=source_sink sync=false";
        
        std::cout << "Creating source pipeline: " << sourcePipelineStr << std::endl;
        sourcePipeline = gst_parse_launch(sourcePipelineStr.c_str(), nullptr);
        if (!sourcePipeline) {
            std::cerr << "Failed to create source pipeline" << std::endl;
            return false;
        }
        
        // 2. PASSTHROUGH PIPELINE: interpipesrc -> encode -> output
        std::string passthroughPipelineStr = 
            "interpipesrc listen-to=source_sink name=passthrough_src is-live=true do-timestamp=true ! "
            "queue max-size-buffers=2 ! "
            "h264parse ! "
            "interpipesink name=passthrough_sink sync=false";

        std::cout << "Creating passthrough pipeline: " << passthroughPipelineStr << std::endl;
        passthroughPipeline = gst_parse_launch(passthroughPipelineStr.c_str(), nullptr);
        if (!passthroughPipeline) {
            std::cerr << "Failed to create passthrough pipeline" << std::endl;
            return false;
        }
        
        // 3. PROCESSING PIPELINE: interpipesrc -> appsink (for receiving frames to process)
        std::string processingPipelineStr = 
            "interpipesrc listen-to=source_sink name=processing_src ! "
            "queue max-size-buffers=2 ! "
            "appsink name=processing_sink sync=false emit-signals=true max-buffers=2 drop=true "
            "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) + 
            ",height=" + std::to_string(frameHeight);
        
        std::cout << "Creating processing pipeline: " << processingPipelineStr << std::endl;
        processingPipeline = gst_parse_launch(processingPipelineStr.c_str(), nullptr);
        if (!processingPipeline) {
            std::cerr << "Failed to create processing pipeline" << std::endl;
            return false;
        }
        
        // 4. PROCESSED OUTPUT PIPELINE: appsrc -> encode -> interpipesink
        std::string processedOutputPipelineStr =
            "appsrc name=processing_appsrc is-live=true format=time block=false max-latency=0 "
            "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) +
            ",height=" + std::to_string(frameHeight) +
            ",framerate=" + std::to_string(int(fps)) + "/1 ! "
            "queue max-size-buffers=2 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate=" + std::to_string(bitrate) + " ! "
            "h264parse ! interpipesink name=processing_sink sync=false";
        
        std::cout << "Creating processed output pipeline: " << processedOutputPipelineStr << std::endl;
        processedOutputPipeline = gst_parse_launch(processedOutputPipelineStr.c_str(), nullptr);
        if (!processedOutputPipeline) {
            std::cerr << "Failed to create processed output pipeline" << std::endl;
            return false;
        }
        
        // 5. OUTPUT PIPELINE: input-selector -> RTSP output
        std::string outputPipelineStr = 
            "input-selector name=switch ! "
            "queue max-size-buffers=2 ! "
            "rtspclientsink location=" + outputAddress + " protocols=tcp latency=0 "
            "interpipesrc listen-to=passthrough_sink name=passthrough_input ! switch.sink_0 "
            "interpipesrc listen-to=processing_sink name=processing_input ! switch.sink_1";
        
        std::cout << "Creating output pipeline: " << outputPipelineStr << std::endl;
        outputPipeline = gst_parse_launch(outputPipelineStr.c_str(), nullptr);
        if (!outputPipeline) {
            std::cerr << "Failed to create output pipeline" << std::endl;
            return false;
        }
        
        // Get elements
        processingAppsink = gst_bin_get_by_name(GST_BIN(processingPipeline), "processing_sink");
        processingAppsrc = gst_bin_get_by_name(GST_BIN(processedOutputPipeline), "processing_appsrc");
        switchElement = gst_bin_get_by_name(GST_BIN(outputPipeline), "switch");
        
        if (!processingAppsink || !processingAppsrc || !switchElement) {
            std::cerr << "Failed to get required elements" << std::endl;
            return false;
        }
        
        // Set up appsink callback
        g_signal_connect(processingAppsink, "new-sample", G_CALLBACK(newSampleCallback), this);
        
        // Set up bus callbacks
        setupBusCallbacks();
        
        // Start all pipelines
        if (!startAllPipelines()) {
            std::cerr << "Failed to start pipelines" << std::endl;
            return false;
        }
        
        // Start processing thread
        processingActive.store(true);
        processingThread = std::thread(&GStreamerPipelineManager::frameProcessingThread, this);
        
        // Start in passthrough mode
        switchToPassthrough();
        
        pipelinesInitialized = true;
        streamInitialized = true;
        
        std::cout << "Interpipe-based seamless switching system initialized successfully!" << std::endl;
        return true;
    }
    
    void setupBusCallbacks() {
        // Set up bus message handling for all pipelines
        GstBus* sourceBus = gst_pipeline_get_bus(GST_PIPELINE(sourcePipeline));
        gst_bus_add_watch(sourceBus, busCallback, this);
        gst_object_unref(sourceBus);
        
        GstBus* passthroughBus = gst_pipeline_get_bus(GST_PIPELINE(passthroughPipeline));
        gst_bus_add_watch(passthroughBus, busCallback, this);
        gst_object_unref(passthroughBus);
        
        GstBus* processingBus = gst_pipeline_get_bus(GST_PIPELINE(processingPipeline));
        gst_bus_add_watch(processingBus, busCallback, this);
        gst_object_unref(processingBus);
        
        GstBus* processedOutputBus = gst_pipeline_get_bus(GST_PIPELINE(processedOutputPipeline));
        gst_bus_add_watch(processedOutputBus, busCallback, this);
        gst_object_unref(processedOutputBus);
        
        GstBus* outputBus = gst_pipeline_get_bus(GST_PIPELINE(outputPipeline));
        gst_bus_add_watch(outputBus, busCallback, this);
        gst_object_unref(outputBus);
    }
    
    bool startAllPipelines() {
        std::cout << "Starting all pipelines..." << std::endl;
        
        // Start source pipeline first
        if (gst_element_set_state(sourcePipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start source pipeline" << std::endl;
            return false;
        }
        
        // Start passthrough pipeline
        if (gst_element_set_state(passthroughPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start passthrough pipeline" << std::endl;
            return false;
        }
        
        // Start processing pipeline
        if (gst_element_set_state(processingPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start processing pipeline" << std::endl;
            return false;
        }
        
        // Start processed output pipeline
        if (gst_element_set_state(processedOutputPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start processed output pipeline" << std::endl;
            return false;
        }
        
        // Start output pipeline
        if (gst_element_set_state(outputPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start output pipeline" << std::endl;
            return false;
        }
        
        // Wait for all pipelines to start
        gst_element_get_state(sourcePipeline, nullptr, nullptr, 5 * GST_SECOND);
        gst_element_get_state(passthroughPipeline, nullptr, nullptr, 5 * GST_SECOND);
        gst_element_get_state(processingPipeline, nullptr, nullptr, 5 * GST_SECOND);
        gst_element_get_state(processedOutputPipeline, nullptr, nullptr, 5 * GST_SECOND);
        gst_element_get_state(outputPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        std::cout << "All pipelines started successfully" << std::endl;
        return true;
    }
    
    bool switchToPassthrough() {
        if (!pipelinesInitialized || !switchElement) {
            std::cerr << "Pipeline not initialized or switch element not found" << std::endl;
            return false;
        }
        
        if (isCurrentlyPassthrough) {
            std::cout << "Already in passthrough mode" << std::endl;
            return true;
        }
        
        std::cout << "Switching to PASSTHROUGH mode..." << std::endl;
        
        // Switch input-selector to passthrough input (sink_0)
        g_object_set(switchElement, "active-pad", 
                     gst_element_get_static_pad(switchElement, "sink_0"), nullptr);
        
        isCurrentlyPassthrough = true;
        std::cout << "Seamlessly switched to PASSTHROUGH mode!" << std::endl;
        return true;
    }
    
    bool switchToProcessing() {
        if (!pipelinesInitialized || !switchElement) {
            std::cerr << "Pipeline not initialized or switch element not found" << std::endl;
            return false;
        }
        
        if (!isCurrentlyPassthrough) {
            std::cout << "Already in processing mode" << std::endl;
            return true;
        }
        
        std::cout << "Switching to PROCESSING mode..." << std::endl;
        
        // Switch input-selector to processing input (sink_1)
        g_object_set(switchElement, "active-pad", 
                     gst_element_get_static_pad(switchElement, "sink_1"), nullptr);
        
        isCurrentlyPassthrough = false;
        std::cout << "Seamlessly switched to PROCESSING mode!" << std::endl;
        return true;
    }
    
    // This method is called from the main thread to process frames
    cv::Mat processFrame(const cv::Mat& frame) {
        // This is where frame processing happens
        // Return the processed frame
        return frame.clone(); // For now, just return a copy
    }
    
    bool pushFrame(const cv::Mat& frame) {
        // This method is no longer used in the interpipe architecture
        // Frames are automatically pulled from the source via appsink callback
        // and processed in the processing thread
        return true;
    }
    
    // Set frame processor function
    void setFrameProcessor(std::function<cv::Mat(const cv::Mat&)> processor) {
        frameProcessor = processor;
    }
    
    void printStatus() {
        if (!pipelinesInitialized) {
            std::cout << "Pipeline Manager Status: Not initialized" << std::endl;
            return;
        }
        
        std::cout << "Interpipe Pipeline Manager Status:" << std::endl;
        std::cout << "  - Source Address: " << sourceAddress << std::endl;
        std::cout << "  - Output Address: " << outputAddress << std::endl;
        std::cout << "  - Resolution: " << frameWidth << "x" << frameHeight << std::endl;
        std::cout << "  - FPS: " << fps << std::endl;
        std::cout << "  - Bitrate: " << bitrate << " Kbps" << std::endl;
        std::cout << "  - Current Mode: " << (isCurrentlyPassthrough ? "Passthrough (Direct)" : "Processing (Enhanced)") << std::endl;
        std::cout << "  - Frame Counter: " << frameCounter << std::endl;
        std::cout << "  - Processing Thread: " << (processingActive.load() ? "Active" : "Inactive") << std::endl;
        std::cout << "  - Frame Queue Size: " << frameQueue.size() << std::endl;
        
        // Print pipeline states
        GstState sourceState, passthroughState, processingState, processedOutputState, outputState;
        if (sourcePipeline) {
            gst_element_get_state(sourcePipeline, &sourceState, nullptr, 0);
            std::cout << "  - Source Pipeline: " << gst_element_state_get_name(sourceState) << std::endl;
        }
        if (passthroughPipeline) {
            gst_element_get_state(passthroughPipeline, &passthroughState, nullptr, 0);
            std::cout << "  - Passthrough Pipeline: " << gst_element_state_get_name(passthroughState) << std::endl;
        }
        if (processingPipeline) {
            gst_element_get_state(processingPipeline, &processingState, nullptr, 0);
            std::cout << "  - Processing Pipeline: " << gst_element_state_get_name(processingState) << std::endl;
        }
        if (processedOutputPipeline) {
            gst_element_get_state(processedOutputPipeline, &processedOutputState, nullptr, 0);
            std::cout << "  - Processed Output Pipeline: " << gst_element_state_get_name(processedOutputState) << std::endl;
        }
        if (outputPipeline) {
            gst_element_get_state(outputPipeline, &outputState, nullptr, 0);
            std::cout << "  - Output Pipeline: " << gst_element_state_get_name(outputState) << std::endl;
        }
    }
    
    void cleanup() {
        std::cout << "Cleaning up interpipe pipeline manager..." << std::endl;
        
        // Stop processing thread
        processingActive.store(false);
        queueCondition.notify_all();
        if (processingThread.joinable()) {
            processingThread.join();
        }
        
        // Stop all pipelines
        if (sourcePipeline) {
            gst_element_set_state(sourcePipeline, GST_STATE_NULL);
            gst_object_unref(sourcePipeline);
            sourcePipeline = nullptr;
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
        
        if (processedOutputPipeline) {
            gst_element_set_state(processedOutputPipeline, GST_STATE_NULL);
            gst_object_unref(processedOutputPipeline);
            processedOutputPipeline = nullptr;
        }
        
        if (outputPipeline) {
            gst_element_set_state(outputPipeline, GST_STATE_NULL);
            gst_object_unref(outputPipeline);
            outputPipeline = nullptr;
        }
        
        // Clean up element references
        if (processingAppsink) {
            gst_object_unref(processingAppsink);
            processingAppsink = nullptr;
        }
        if (processingAppsrc) {
            gst_object_unref(processingAppsrc);
            processingAppsrc = nullptr;
        }
        if (switchElement) {
            gst_object_unref(switchElement);
            switchElement = nullptr;
        }
        
        pipelinesInitialized = false;
        streamInitialized = false;
        
        std::cout << "Interpipe pipeline manager cleanup complete" << std::endl;
    }
    
    ~GStreamerPipelineManager() {
        cleanup();
    }
    
private:
    // Frame processor function - can be set from outside
    std::function<cv::Mat(const cv::Mat&)> frameProcessor;
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

    gst_init(&argc, &argv);

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

    // Control variables for runtime monitoring
    int emptyFrameCount = 0;
    int configCheckCounter = 0;

    // Initialize tracker and TCP receiver for tracking coordinates
    std::unique_ptr<vs::DeepStreamTracker> tracker = std::make_unique<vs::DeepStreamTracker>(trackerParams);
    vs::TcpReciever tcp(5000);   // listen on port 5000
    tcp.start();

    // Frame dimensions from config or defaults
    int frameWidth = runParams.width > 0 ? runParams.width : 1920;
    int frameHeight = runParams.height > 0 ? runParams.height : 1080;
    double fps = 30.0; // Default FPS, GStreamer will auto-detect from source
    
    std::cout << "Target frame dimensions: " << frameWidth << "x" << frameHeight << std::endl;

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
    
    // Set up frame processor function for the pipeline manager
    pipelineManager.setFrameProcessor([&](const cv::Mat& frame) -> cv::Mat {
        cv::Mat processedFrame = frame.clone();
        
        try {
            // Apply processing steps based on configuration
            if (runParams.enhancerEnabled) {
                processedFrame = vs::Enhancer::enhanceImage(processedFrame, enhancerParams);
            }

            // Apply Roll Correction
            if (runParams.rollCorrectionEnabled) {
                processedFrame = vs::RollCorrection::autoCorrectRoll(processedFrame, rollParams);
            }

            // Apply Stabilization
            if (runParams.stabilizationEnabled) {
                cv::Mat stabilizedFrame = stab.stabilize(processedFrame);
                if (!stabilizedFrame.empty()) {
                    processedFrame = stabilizedFrame;
                }
            }

            // Apply tracking if enabled
            if (runParams.trackerEnabled) {
                // Get tracking coordinates from TCP
                int x, y;
                if (tcp.tryGetLatest(x, y)) {
                    // Process frame through tracker
                    auto detections = tracker->processFrame(processedFrame);
                    
                    // Draw detections with the selected coordinates
                    if (x >= 0 && y >= 0) {
                        processedFrame = tracker->drawDetections(processedFrame, detections, x, y);
                    } else {
                        processedFrame = tracker->drawDetections(processedFrame, detections);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Frame processing error: " << e.what() << std::endl;
            // Return original frame on error
            return frame.clone();
        }
        
        return processedFrame;
    });
    
    if (!pipelineManager.initialize(frameWidth, frameHeight, fps)) {
        std::cerr << "Failed to initialize pipeline manager" << std::endl;
        return -1;
    }

    // Set initial mode based on config
    if (usePassthrough) {
        pipelineManager.switchToPassthrough();
        std::cout << "Starting in PASSTHROUGH mode - seamless direct RTSP forwarding" << std::endl;
    } else {
        pipelineManager.switchToProcessing();
        std::cout << "Starting in PROCESSING mode - seamless frame processing" << std::endl;
    }

    std::cout << "Interpipe-based seamless switching system is active!" << std::endl;

    // Create display windows if needed
    const int windowWidth = runParams.width;
    const int windowHeight = runParams.height;
    
    if (!runParams.optimizeFps) {
        cv::namedWindow("Raw", cv::WINDOW_NORMAL);
        cv::resizeWindow("Raw", windowWidth, windowHeight);
        cv::namedWindow("Final", cv::WINDOW_NORMAL);
        cv::resizeWindow("Final", windowWidth, windowHeight);
    }

    // Main monitoring loop - much simpler now since processing is handled by GStreamer
    int loopCounter = 0;
    while (!stopRequested) {
        // Configuration file monitoring
        if (loopCounter % 30 == 0) {  // Check every 30 loops
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
                        
                        // Switch mode if needed - seamlessly!
                        if (oldPassthrough != usePassthrough) {
                            if (usePassthrough) {
                                pipelineManager.switchToPassthrough();
                                std::cout << "Seamlessly switched to PASSTHROUGH mode" << std::endl;
                            } else {
                                pipelineManager.switchToProcessing();
                                std::cout << "Seamlessly switched to PROCESSING mode" << std::endl;
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

        // Print performance stats occasionally
        if (loopCounter % 300 == 0) {  // Every 300 loops (~30 seconds)
            std::cout << "Loop counter: " << loopCounter << " | Mode: " << 
                      (usePassthrough ? "Passthrough" : "Processing") << std::endl;
            
            pipelineManager.printStatus();
        }

        // Handle keyboard input for manual switching
        if (!runParams.optimizeFps && loopCounter % 10 == 0) {
            int key = cv::waitKey(1) & 0xFF;
            
            switch (key) {
                case 'p':
                    usePassthrough = true;
                    pipelineManager.switchToPassthrough();
                    std::cout << "Manually switched to PASSTHROUGH mode (seamless)" << std::endl;
                    break;
                    
                case 'r':
                    usePassthrough = false;
                    pipelineManager.switchToProcessing();
                    std::cout << "Manually switched to PROCESSING mode (seamless)" << std::endl;
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
        
        loopCounter++;
        
        // Sleep for 100ms - no need for tight timing since GStreamer handles everything
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "Shutting down interpipe system..." << std::endl;
    
    // Stop TCP receiver
    tcp.stop();
    
    // Cleanup pipeline manager
    pipelineManager.cleanup();
    
    cv::destroyAllWindows();
    std::cout << "Interpipe system shutdown complete." << std::endl;
    return 0;
}
