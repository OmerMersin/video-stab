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
#include <queue>   // For std::queue
#include <atomic>  // For std::atomic
#include <mutex>   // For std::mutex
#include <condition_variable>  // For std::condition_variable
#include <functional>  // For std::function
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
    
    // Low-level interpipe-based seamless switching architecture
    GstElement* sourcePipeline;          // RTSP source to interpipesink
    GstElement* passthroughPipeline;     // interpipesrc -> interpipesink (passthrough frames)
    GstElement* processingPipeline;      // interpipesrc -> appsink (for receiving frames)
    GstElement* processedOutputPipeline; // appsrc -> interpipesink (for processed frames)
    GstElement* outputPipeline;          // Persistent output pipeline that NEVER gets destroyed
    
    // Processing components
    GstElement* processingAppsink;    // Receives frames for processing
    GstElement* processingAppsrc;     // Sends processed frames
    GstElement* inputSelector;        // Switches between passthrough and processing modes
    GstElement* outputSource;         // interpipesrc in output pipeline for switching
    
    // Interpipe elements for explicit control
    GstElement* sourceInterpipeSink;     // source pipeline interpipesink
    GstElement* passthroughInterpipeSink; // passthrough pipeline interpipesink
    GstElement* processedInterpipeSink;   // processed output pipeline interpipesink
    
    // GMainLoop for GStreamer events
    GMainLoop* mainLoop;
    std::thread gstLoopThread;
    std::atomic<bool> loopActive;
    
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
            std::cerr << "Cannot push frame: appsrc=" << (processingAppsrc ? "valid" : "null") 
                      << ", frame.empty=" << frame.empty() << std::endl;
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
        
        // Debug output every 30 frames
        if (frameCounter % 30 == 0) {
            std::cout << "✅ Pushed frame #" << frameCounter << " (" << frame.cols << "x" << frame.rows 
                      << ") to processing output pipeline" << std::endl;
            
            // Check pipeline state
            GstState state;
            gst_element_get_state(GST_ELEMENT(processingAppsrc), &state, nullptr, 0);
            std::cout << "  - ProcessingAppsrc state: " << gst_element_state_get_name(state) << std::endl;
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
          processingAppsrc(nullptr), inputSelector(nullptr),
          sourceInterpipeSink(nullptr), passthroughInterpipeSink(nullptr), processedInterpipeSink(nullptr),
          mainLoop(nullptr), loopActive(false),
          streamInitialized(false), frameCounter(0), 
          isCurrentlyPassthrough(true), processingActive(false) {
        
        // Initialize GStreamer
        if (!gst_is_initialized()) {
            gst_init(nullptr, nullptr);
        }
        
        // Create main loop for GStreamer events
        mainLoop = g_main_loop_new(nullptr, FALSE);
    }
    
    bool initialize(int width, int height, double framerate) {
        frameWidth = width;
        frameHeight = height;
        fps = framerate;
        
        if (streamInitialized) return true;
        
        std::cout << "Initializing low-level interpipe-based seamless switching system..." << std::endl;
        
        // Calculate bitrate
        int calculatedBitrate = std::max(2000000, std::min(8000000, static_cast<int>(frameWidth * frameHeight * fps * 0.1)));
        bitrate = calculatedBitrate;
        
        // 1. SOURCE PIPELINE: RTSP source -> interpipesink (exactly like test_interpipe)
        sourcePipeline = gst_parse_launch(
            ("rtspsrc location=" + sourceAddress + " latency=0 ! "
             "rtph264depay ! h264parse ! "
             "interpipesink name=source_sink sync=false async=false").c_str(), 
            nullptr);
        
        if (!sourcePipeline) {
            std::cerr << "Failed to create source pipeline" << std::endl;
            return false;
        }
        
        // Get interpipesink element and set explicit caps like gstd does
        sourceInterpipeSink = gst_bin_get_by_name(GST_BIN(sourcePipeline), "source_sink");
        if (sourceInterpipeSink) {
            GstCaps* h264Caps = gst_caps_from_string("video/x-h264,stream-format=byte-stream,alignment=au");
            g_object_set(sourceInterpipeSink, "caps", h264Caps, nullptr);
            gst_caps_unref(h264Caps);
            std::cout << "Set explicit H264 caps on source interpipesink" << std::endl;
        }
        
        // 2. PASSTHROUGH PIPELINE: interpipesrc -> direct output (ZERO LATENCY - no queues/buffers)
        passthroughPipeline = gst_parse_launch(
            "interpipesrc listen-to=source_sink name=passthrough_src is-live=true format=time ! "
            "interpipesink name=passthrough_sink sync=false async=false",
            nullptr);
        
        if (!passthroughPipeline) {
            std::cerr << "Failed to create passthrough pipeline" << std::endl;
            return false;
        }
        
        // Get passthrough interpipesink and set caps
        passthroughInterpipeSink = gst_bin_get_by_name(GST_BIN(passthroughPipeline), "passthrough_sink");
        if (passthroughInterpipeSink) {
            GstCaps* h264Caps = gst_caps_from_string("video/x-h264,stream-format=byte-stream,alignment=au");
            g_object_set(passthroughInterpipeSink, "caps", h264Caps, nullptr);
            gst_caps_unref(h264Caps);
            std::cout << "Set explicit H264 caps on passthrough interpipesink" << std::endl;
        }
        
        // 3. PROCESSING PIPELINE: interpipesrc -> decode -> appsink (for receiving frames to process)
        processingPipeline = gst_parse_launch(
            "interpipesrc listen-to=source_sink name=processing_src is-live=true format=time ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            "h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR !"
            "appsink name=processing_sink sync=false emit-signals=true max-buffers=2 drop=true",
            nullptr);

        if (!processingPipeline) {
            std::cerr << "Failed to create processing pipeline" << std::endl;
            return false;
        }
        
        // 4. PROCESSED OUTPUT PIPELINE: appsrc -> encode -> interpipesink (for processed frames)
        processedOutputPipeline = gst_parse_launch(
            ("appsrc name=processing_appsrc is-live=true format=time block=false max-latency=0 "
             "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) +
             ",height=" + std::to_string(frameHeight) +
             ",framerate=" + std::to_string(int(fps)) + "/1 ! "
             "queue max-size-buffers=2 leaky=downstream ! "
             "videoconvert ! video/x-raw,format=I420 ! "
             "x264enc tune=zerolatency speed-preset=superfast bitrate=" + std::to_string(bitrate/1000) + " key-int-max=30 ! "
             "h264parse ! "
             "interpipesink name=processed_sink sync=false async=false").c_str(),
            nullptr);
        
        if (!processedOutputPipeline) {
            std::cerr << "Failed to create processed output pipeline" << std::endl;
            return false;
        }
        
        // Get processed interpipesink and set caps
        processedInterpipeSink = gst_bin_get_by_name(GST_BIN(processedOutputPipeline), "processed_sink");
        if (processedInterpipeSink) {
            GstCaps* h264Caps = gst_caps_from_string("video/x-h264,stream-format=byte-stream,alignment=au");
            g_object_set(processedInterpipeSink, "caps", h264Caps, nullptr);
            gst_caps_unref(h264Caps);
            std::cout << "Set explicit H264 caps on processed interpipesink" << std::endl;
        }
        
        // 5. SIMPLE OUTPUT PIPELINE: interpipesrc -> RTSP (starts with passthrough)
        outputPipeline = gst_parse_launch(
            ("interpipesrc listen-to=passthrough_sink name=output_source is-live=true format=time ! "
             "queue max-size-buffers=1 leaky=downstream ! "
             "rtspclientsink location=" + outputAddress + " protocols=tcp latency=0").c_str(),
            nullptr);
        
        if (!outputPipeline) {
            std::cerr << "Failed to create output pipeline" << std::endl;
            return false;
        }
        
        // Get the interpipesrc element for switching
        outputSource = gst_bin_get_by_name(GST_BIN(outputPipeline), "output_source");
        if (!outputSource) {
            std::cerr << "Failed to get output source element" << std::endl;
            return false;
        }
        
        // Get elements for runtime control
        processingAppsink = gst_bin_get_by_name(GST_BIN(processingPipeline), "processing_sink");
        processingAppsrc = gst_bin_get_by_name(GST_BIN(processedOutputPipeline), "processing_appsrc");
        
        if (!processingAppsink || !processingAppsrc || !outputSource) {
            std::cerr << "Failed to get required elements" << std::endl;
            return false;
        }
        
        // Set up appsink callback for frame processing
        g_signal_connect(processingAppsink, "new-sample", G_CALLBACK(newSampleCallback), this);
        
        // Set up bus callbacks for all pipelines
        setupBusCallbacks();
        
        // Start GStreamer main loop in separate thread
        loopActive.store(true);
        gstLoopThread = std::thread([this]() {
            std::cout << "Starting GStreamer main loop" << std::endl;
            g_main_loop_run(mainLoop);
            std::cout << "GStreamer main loop stopped" << std::endl;
        });
        
        // Set pipelines as initialized BEFORE starting them
        pipelinesInitialized = true;
        
        // Start processing thread
        processingActive.store(true);
        processingThread = std::thread(&GStreamerPipelineManager::frameProcessingThread, this);
        
        // Start pipelines for initial passthrough mode
        std::cout << "Starting pipelines for initial passthrough mode..." << std::endl;
        gst_element_set_state(sourcePipeline, GST_STATE_PLAYING);
        gst_element_set_state(passthroughPipeline, GST_STATE_PLAYING);
        gst_element_set_state(outputPipeline, GST_STATE_PLAYING);
        
        // Keep processing pipelines in NULL state initially
        gst_element_set_state(processingPipeline, GST_STATE_NULL);
        gst_element_set_state(processedOutputPipeline, GST_STATE_NULL);
        
        streamInitialized = true;
        isCurrentlyPassthrough = true;
        
        std::cout << "✅ Pipeline system initialized in PASSTHROUGH mode" << std::endl;
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
    
    
    bool switchToPassthrough() {
        if (!pipelinesInitialized) {
            std::cerr << "Pipelines not initialized" << std::endl;
            return false;
        }
        
        if (isCurrentlyPassthrough) {
            std::cout << "Already in passthrough mode" << std::endl;
            return true;
        }
        
        std::cout << "=== SWITCHING TO PASSTHROUGH MODE ===" << std::endl;
        
        // Start passthrough pipeline FIRST
        std::cout << "Starting passthrough pipeline..." << std::endl;
        if (gst_element_set_state(passthroughPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start passthrough pipeline" << std::endl;
            return false;
        }
        
        // Wait for passthrough pipeline to stabilize
        std::cout << "Waiting for passthrough pipeline to stabilize..." << std::endl;
        g_usleep(300000); // 300ms
        
        // Switch output source to listen to passthrough_sink
        g_object_set(outputSource, "listen-to", "passthrough_sink", NULL);
        std::cout << "Switched output source to passthrough_sink" << std::endl;
        
        // Wait a bit for the switch to take effect
        g_usleep(200000); // 200ms
        
        // Stop processing pipelines to save resources
        gst_element_set_state(processingPipeline, GST_STATE_NULL);
        gst_element_set_state(processedOutputPipeline, GST_STATE_NULL);
        
        isCurrentlyPassthrough = true;
        
        std::cout << "✅ Successfully switched to ZERO-LATENCY passthrough mode" << std::endl;
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
        
        std::cout << "=== SWITCHING TO PROCESSING MODE ===" << std::endl;
        
        // Start processing pipelines FIRST
        std::cout << "Starting processing pipeline..." << std::endl;
        if (gst_element_set_state(processingPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start processing pipeline" << std::endl;
            return false;
        }
        
        std::cout << "Starting processed output pipeline..." << std::endl;
        if (gst_element_set_state(processedOutputPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start processed output pipeline" << std::endl;
            return false;
        }
        
        // Wait for processing pipelines to fully start and stabilize
        std::cout << "Waiting for processing pipelines to stabilize..." << std::endl;
        g_usleep(500000); // 500ms - longer wait for proper startup
        
        // Ensure processing pipeline is actually PLAYING
        GstState state;
        gst_element_get_state(processingPipeline, &state, nullptr, GST_CLOCK_TIME_NONE);
        if (state != GST_STATE_PLAYING) {
            std::cerr << "Processing pipeline failed to reach PLAYING state: " << gst_element_state_get_name(state) << std::endl;
            return false;
        }
        
        gst_element_get_state(processedOutputPipeline, &state, nullptr, GST_CLOCK_TIME_NONE);
        if (state != GST_STATE_PLAYING) {
            std::cerr << "Processed output pipeline failed to reach PLAYING state: " << gst_element_state_get_name(state) << std::endl;
            return false;
        }
        
        std::cout << "Both processing pipelines confirmed PLAYING" << std::endl;
        
        // Switch output source to listen to processed_sink instead of passthrough_sink
        g_object_set(outputSource, "listen-to", "processed_sink", NULL);
        std::cout << "Switched output source to processed_sink" << std::endl;
        
        // Wait a bit for the switch to take effect
        g_usleep(200000); // 200ms
        
        // Stop passthrough pipeline to save resources
        gst_element_set_state(passthroughPipeline, GST_STATE_NULL);
        
        isCurrentlyPassthrough = false;
        
        std::cout << "✅ Successfully switched to processing mode" << std::endl;
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
            std::cout << "  - Processed Output Pipeline (Direct RTSP): " << gst_element_state_get_name(processedOutputState) << std::endl;
        }
        if (outputPipeline) {
            gst_element_get_state(outputPipeline, &outputState, nullptr, 0);
            std::cout << "  - Output Pipeline: " << gst_element_state_get_name(outputState) << std::endl;
        } else if (!isCurrentlyPassthrough) {
            std::cout << "  - Output Pipeline: Using direct RTSP (processedOutputPipeline)" << std::endl;
        }
    }
    
    void cleanup() {
        std::cout << "Cleaning up low-level interpipe pipeline manager..." << std::endl;
        
        // Stop GStreamer main loop
        if (loopActive.load()) {
            loopActive.store(false);
            if (mainLoop) {
                g_main_loop_quit(mainLoop);
            }
            if (gstLoopThread.joinable()) {
                gstLoopThread.join();
            }
        }
        
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
        if (inputSelector) {
            gst_object_unref(inputSelector);
            inputSelector = nullptr;
        }
        if (sourceInterpipeSink) {
            gst_object_unref(sourceInterpipeSink);
            sourceInterpipeSink = nullptr;
        }
        if (passthroughInterpipeSink) {
            gst_object_unref(passthroughInterpipeSink);
            passthroughInterpipeSink = nullptr;
        }
        if (processedInterpipeSink) {
            gst_object_unref(processedInterpipeSink);
            processedInterpipeSink = nullptr;
        }
        
        // Clean up main loop
        if (mainLoop) {
            g_main_loop_unref(mainLoop);
            mainLoop = nullptr;
        }
        
        pipelinesInitialized = false;
        streamInitialized = false;
        
        std::cout << "Low-level interpipe pipeline manager cleanup complete" << std::endl;
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

    // Read Stabilizer Parameters - Professional Grade (iPhone/GoPro-like)
    cv::FileNode stabNode = fs["stabilizer"];
    if (!stabNode.empty()) {
        // Basic parameters
        stabNode["smoothing_radius"] >> stabParams.smoothingRadius;
        stabNode["border_type"] >> stabParams.borderType;
        stabNode["border_size"] >> stabParams.borderSize;
        stabNode["crop_n_zoom"] >> stabParams.cropNZoom;
        stabNode["logging"] >> stabParams.logging;
        stabNode["use_cuda"] >> stabParams.useCuda;
        
        // Professional smoothing methods
        stabNode["smoothing_method"] >> stabParams.smoothingMethod;
        stabNode["gaussian_sigma"] >> stabParams.gaussianSigma;
        
        // Multi-stage smoothing (Sightline-inspired)
        stabNode["stage_one_radius"] >> stabParams.stageOneRadius;
        stabNode["stage_two_radius"] >> stabParams.stageTwoRadius;
        stabNode["use_temporal_filtering"] >> stabParams.useTemporalFiltering;
        stabNode["temporal_window_size"] >> stabParams.temporalWindowSize;
        
        // Adaptive smoothing
        stabNode["adaptive_smoothing"] >> stabParams.adaptiveSmoothing;
        stabNode["min_smoothing_radius"] >> stabParams.minSmoothingRadius;
        stabNode["max_smoothing_radius"] >> stabParams.maxSmoothingRadius;
        
        // Feature detection parameters
        stabNode["max_corners"] >> stabParams.maxCorners;
        stabNode["quality_level"] >> stabParams.qualityLevel;
        stabNode["min_distance"] >> stabParams.minDistance;
        stabNode["block_size"] >> stabParams.blockSize;
        
        // Motion analysis
        stabNode["outlier_threshold"] >> stabParams.outlierThreshold;
        stabNode["motion_prediction"] >> stabParams.motionPrediction;
        stabNode["intentional_motion_threshold"] >> stabParams.intentionalMotionThreshold;
        
        // Professional jitter filtering
        int jitterFreq = 3; // Default to ADAPTIVE
        stabNode["jitter_frequency"] >> jitterFreq;
        stabParams.jitterFrequency = static_cast<vs::Stabilizer::Parameters::JitterFrequency>(jitterFreq);
        stabNode["separate_translation_rotation"] >> stabParams.separateTranslationRotation;
        
        // Advanced features
        stabNode["deep_stabilization"] >> stabParams.deepStabilization;
        stabNode["model_path"] >> stabParams.modelPath;
        stabNode["roll_compensation"] >> stabParams.rollCompensation;
        stabNode["roll_compensation_factor"] >> stabParams.rollCompensationFactor;
        
        // Region of Interest
        stabNode["use_roi"] >> stabParams.useROI;
        if (stabParams.useROI) {
            int roiX = 0, roiY = 0, roiW = 0, roiH = 0;
            stabNode["roi_x"] >> roiX;
            stabNode["roi_y"] >> roiY;
            stabNode["roi_width"] >> roiW;
            stabNode["roi_height"] >> roiH;
            stabParams.roi = cv::Rect(roiX, roiY, roiW, roiH);
        }
        
        // Horizon lock
        stabNode["horizon_lock"] >> stabParams.horizonLock;
        
        // Feature detector selection
        int featureDetectorType = 0; // Default to GFTT
        stabNode["feature_detector_type"] >> featureDetectorType;
        stabParams.featureDetector = static_cast<vs::Stabilizer::Parameters::FeatureDetector>(featureDetectorType);
        stabNode["fast_threshold"] >> stabParams.fastThreshold;
        stabNode["orb_features"] >> stabParams.orbFeatures;
        
        // Enhanced border effects
        stabNode["border_scale_factor"] >> stabParams.borderScaleFactor;
        stabNode["motion_threshold_low"] >> stabParams.motionThresholdLow;
        stabNode["motion_threshold_high"] >> stabParams.motionThresholdHigh;
        
        // Fade parameters
        stabNode["fadeDuration"] >> stabParams.fadeDuration;
        stabNode["fadeAlpha"] >> stabParams.fadeAlpha;
        
        // Additional professional parameters (missing from previous config)
        stabNode["use_imu_data"] >> stabParams.useImuData;
        
        // Virtual Canvas Stabilization parameters
        stabNode["enable_virtual_canvas"] >> stabParams.enableVirtualCanvas;
        stabNode["canvas_scale_factor"] >> stabParams.canvasScaleFactor;
        stabNode["temporal_buffer_size"] >> stabParams.temporalBufferSize;
        stabNode["canvas_blend_weight"] >> stabParams.canvasBlendWeight;
        stabNode["adaptive_canvas_size"] >> stabParams.adaptiveCanvasSize;
        stabNode["max_canvas_scale"] >> stabParams.maxCanvasScale;
        stabNode["min_canvas_scale"] >> stabParams.minCanvasScale;
        stabNode["preserve_edge_quality"] >> stabParams.preserveEdgeQuality;
        stabNode["edge_blend_radius"] >> stabParams.edgeBlendRadius;
        
        // HF: Drone high-frequency vibration suppression parameters
        stabNode["drone_high_freq_mode"] >> stabParams.droneHighFreqMode;
        stabNode["hf_shake_px"] >> stabParams.hfShakePx;
        stabNode["hf_analysis_max_width"] >> stabParams.hfAnalysisMaxWidth;
        stabNode["hf_rot_lp_alpha"] >> stabParams.hfRotLPAlpha;
        stabNode["enable_conditional_clahe"] >> stabParams.enableConditionalCLAHE;
        
        // HF: Dead zone parameters for freeze shot
        stabNode["hf_dead_zone_threshold"] >> stabParams.hfDeadZoneThreshold;
        stabNode["hf_freeze_duration"] >> stabParams.hfFreezeDuration;
        stabNode["hf_motion_accumulator_decay"] >> stabParams.hfMotionAccumulatorDecay;
        
        // Professional motion classification thresholds
        float shakeThreshold = 3.0f, walkingThreshold = 8.0f, vehicleThreshold = 15.0f;
        stabNode["shake_level_threshold"] >> shakeThreshold;
        stabNode["walking_detection_threshold"] >> walkingThreshold;
        stabNode["vehicle_detection_threshold"] >> vehicleThreshold;
        // Note: These are used internally by the stabilizer for motion classification
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
    std::cout << "Enhancer: " << (runParams.enhancerEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Roll Correction: " << (runParams.rollCorrectionEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Stabilizer: " << (runParams.stabilizationEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Tracker: " << (runParams.trackerEnabled ? "Enabled" : "Disabled") << std::endl;

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

    std::cout << "Determined mode: " << (usePassthrough ? "PASSTHROUGH" : "PROCESSING") << std::endl;

    // Create and initialize pipeline manager with source and output addresses
    std::string sourceAddress = videoSource;  // Use video source from config
    std::string outputAddress = "rtsp://192.168.144.150:8554/forwarded";  // RTSP output
    int bitrate = std::max(2000000, std::min(8000000, static_cast<int>(frameWidth * frameHeight * fps * 0.1)));
    
    GStreamerPipelineManager pipelineManager(sourceAddress, outputAddress, bitrate);
    
    // Create stabilizer as smart pointer so we can recreate it
    std::unique_ptr<vs::Stabilizer> stab = std::make_unique<vs::Stabilizer>(stabParams);

    // Set up frame processor to use the smart pointer
    pipelineManager.setFrameProcessor([&](const cv::Mat& frame) -> cv::Mat {
        cv::Mat processedFrame = frame.clone();
        
        // Debug counter to verify parameter usage
        static int debugCounter = 0;
        
        try {
            // Apply processing steps based on configuration
            if (runParams.enhancerEnabled) {
                processedFrame = vs::Enhancer::enhanceImage(processedFrame, enhancerParams);
            }

            // Apply Roll Correction
            if (runParams.rollCorrectionEnabled) {
                processedFrame = vs::RollCorrection::autoCorrectRoll(processedFrame, rollParams);
            }

            // Apply Stabilization with current parameters
            if (runParams.stabilizationEnabled && stab) {
                // Debug output every 300 frames to verify current parameters
                if ((debugCounter++ % 300) == 0) {
                    std::cout << "🎯 Current stabilizer mode: " << stabParams.smoothingMethod 
                              << ", radius: " << stabParams.smoothingRadius 
                              << ", horizon_lock: " << (stabParams.horizonLock ? "ON" : "OFF") << std::endl;
                }
                
                cv::Mat stabilizedFrame = stab->stabilize(processedFrame);
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
            bool oldStabilizerEnabled = runParams.stabilizationEnabled;
            
            if (readConfig(configFile, videoSource, runParams, enhancerParams, rollParams, stabParams, camParams, trackerParams)) {
                // Update processing mode
                usePassthrough = !runParams.enhancerEnabled && 
                               !runParams.rollCorrectionEnabled && 
                               !runParams.stabilizationEnabled &&
                               !runParams.trackerEnabled;
                
                // IMPORTANT: Recreate stabilizer with new parameters
                if (runParams.stabilizationEnabled) {
                    std::cout << "Recreating stabilizer with new parameters..." << std::endl;
                    std::cout << "  - Smoothing radius: " << stabParams.smoothingRadius << std::endl;
                    std::cout << "  - Smoothing method: " << stabParams.smoothingMethod << std::endl;
                    std::cout << "  - Gaussian sigma: " << stabParams.gaussianSigma << std::endl;
                    std::cout << "  - Intentional motion threshold: " << stabParams.intentionalMotionThreshold << std::endl;
                    std::cout << "  - Horizon lock: " << (stabParams.horizonLock ? "Enabled" : "Disabled") << std::endl;
                    std::cout << "  - Adaptive smoothing: " << (stabParams.adaptiveSmoothing ? "Enabled" : "Disabled") << std::endl;
                    
                    // Clean up old stabilizer
                    if (stab) {
                        stab->clean();
                    }
                    
                    // Create new stabilizer with updated parameters
                    stab = std::make_unique<vs::Stabilizer>(stabParams);
                    std::cout << "Stabilizer recreated successfully!" << std::endl;
                } else if (oldStabilizerEnabled && !runParams.stabilizationEnabled) {
                    // Stabilizer was disabled
                    std::cout << "Stabilizer disabled, cleaning up..." << std::endl;
                    if (stab) {
                        stab->clean();
                        stab.reset();
                    }
                }
                
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
