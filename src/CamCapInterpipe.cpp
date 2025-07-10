#include "video/CamCapInterpipe.h"
#include <iostream>
#include <stdexcept>
#include <chrono>

namespace vs {

CamCapInterpipe::CamCapInterpipe(const Parameters& params) 
    : params(params), inputPipeline(nullptr), outputPipeline(nullptr), 
      appSink(nullptr), appSrc(nullptr) {
    
    if (params.logging) {
        std::cout << "[CamCapInterpipe] Initializing with interpipe input: " 
                  << params.interpipeInputName << std::endl;
    }
}

CamCapInterpipe::~CamCapInterpipe() {
    stop();
}

bool CamCapInterpipe::initialize() {
    if (initialized) {
        return true;
    }

    gst_init(nullptr, nullptr);

    // Create input pipeline to receive frames from interpipe
    std::string inputPipelineStr = 
        "interpipesrc listen-to=" + params.interpipeInputName + 
        " is-live=true format=time ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "appsink name=sink sync=false async=false drop=true max-buffers=1";

    if (params.logging) {
        std::cout << "[CamCapInterpipe] Input pipeline: " << inputPipelineStr << std::endl;
    }

    inputPipeline = gst_parse_launch(inputPipelineStr.c_str(), nullptr);
    if (!inputPipeline) {
        std::cerr << "[CamCapInterpipe] Failed to create input pipeline" << std::endl;
        return false;
    }

    // Get appsink
    appSink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(inputPipeline), "sink"));
    if (!appSink) {
        std::cerr << "[CamCapInterpipe] Failed to get appsink" << std::endl;
        return false;
    }

    // Configure appsink
    GstCaps* caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, params.width,
        "height", G_TYPE_INT, params.height,
        "framerate", GST_TYPE_FRACTION, params.fps, 1,
        nullptr);
    
    gst_app_sink_set_caps(appSink, caps);
    gst_caps_unref(caps);

    // Set callback for new samples
    GstAppSinkCallbacks callbacks = {0};
    callbacks.new_sample = newSampleCallback;
    gst_app_sink_set_callbacks(appSink, &callbacks, this, nullptr);

    // Create output pipeline to send processed frames
    std::string outputPipelineStr = 
        "appsrc name=src is-live=true format=time block=false max-latency=0 "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(params.width) + 
        ",height=" + std::to_string(params.height) + 
        ",framerate=" + std::to_string(params.fps) + "/1 ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "x264enc tune=zerolatency speed-preset=superfast bitrate=4000 key-int-max=30 ! "
        "h264parse ! "
        "interpipesink name=" + params.interpipeOutputName + " sync=false async=false";

    if (params.logging) {
        std::cout << "[CamCapInterpipe] Output pipeline: " << outputPipelineStr << std::endl;
    }

    outputPipeline = gst_parse_launch(outputPipelineStr.c_str(), nullptr);
    if (!outputPipeline) {
        std::cerr << "[CamCapInterpipe] Failed to create output pipeline" << std::endl;
        return false;
    }

    // Get appsrc
    appSrc = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(outputPipeline), "src"));
    if (!appSrc) {
        std::cerr << "[CamCapInterpipe] Failed to get appsrc" << std::endl;
        return false;
    }

    // Configure appsrc
    g_object_set(appSrc,
        "is-live", TRUE,
        "block", FALSE,
        "format", GST_FORMAT_TIME,
        "max-latency", G_GINT64_CONSTANT(0),
        "do-timestamp", TRUE,
        nullptr);

    // Set bus callbacks
    GstBus* inputBus = gst_element_get_bus(inputPipeline);
    gst_bus_add_watch(inputBus, busCallback, this);
    gst_object_unref(inputBus);

    GstBus* outputBus = gst_element_get_bus(outputPipeline);
    gst_bus_add_watch(outputBus, busCallback, this);
    gst_object_unref(outputBus);

    initialized = true;
    
    if (params.logging) {
        std::cout << "[CamCapInterpipe] Initialization complete" << std::endl;
    }

    return true;
}

void CamCapInterpipe::start() {
    if (!initialized) {
        std::cerr << "[CamCapInterpipe] Not initialized" << std::endl;
        return;
    }

    if (isRunning) {
        return;
    }

    terminate = false;
    isRunning = true;

    // Start pipelines
    gst_element_set_state(inputPipeline, GST_STATE_PLAYING);
    gst_element_set_state(outputPipeline, GST_STATE_PLAYING);

    // Start threads
    inputThread = std::thread(&CamCapInterpipe::inputLoop, this);
    outputThread = std::thread(&CamCapInterpipe::outputLoop, this);

    if (params.logging) {
        std::cout << "[CamCapInterpipe] Started" << std::endl;
    }
}

void CamCapInterpipe::stop() {
    if (!isRunning) {
        return;
    }

    terminate = true;
    isRunning = false;

    // Notify threads
    {
        std::lock_guard<std::mutex> lock(inputQueueMutex);
        inputQueueCondition.notify_all();
    }
    {
        std::lock_guard<std::mutex> lock(outputQueueMutex);
        outputQueueCondition.notify_all();
    }

    // Join threads
    if (inputThread.joinable()) {
        inputThread.join();
    }
    if (outputThread.joinable()) {
        outputThread.join();
    }

    // Stop pipelines
    if (inputPipeline) {
        gst_element_set_state(inputPipeline, GST_STATE_NULL);
        gst_object_unref(inputPipeline);
        inputPipeline = nullptr;
    }

    if (outputPipeline) {
        gst_element_set_state(outputPipeline, GST_STATE_NULL);
        gst_object_unref(outputPipeline);
        outputPipeline = nullptr;
    }

    // Clear queues
    {
        std::lock_guard<std::mutex> lock(inputQueueMutex);
        while (!inputFrameQueue.empty()) {
            inputFrameQueue.pop();
        }
    }
    {
        std::lock_guard<std::mutex> lock(outputQueueMutex);
        while (!outputFrameQueue.empty()) {
            outputFrameQueue.pop();
        }
    }

    if (params.logging) {
        std::cout << "[CamCapInterpipe] Stopped" << std::endl;
    }
}

cv::Mat CamCapInterpipe::read() {
    if (!isRunning) {
        return cv::Mat();
    }

    std::unique_lock<std::mutex> lock(inputQueueMutex);
    
    auto startTime = std::chrono::steady_clock::now();
    
    while (inputFrameQueue.empty() && !terminate) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime).count();
        
        if (elapsed > params.threadTimeout) {
            if (params.logging) {
                std::cerr << "[CamCapInterpipe] Timeout waiting for frame" << std::endl;
            }
            return cv::Mat();
        }
        
        inputQueueCondition.wait_for(lock, std::chrono::milliseconds(1));
    }

    if (terminate || inputFrameQueue.empty()) {
        return cv::Mat();
    }

    cv::Mat frame = inputFrameQueue.front();
    inputFrameQueue.pop();
    
    return frame;
}

void CamCapInterpipe::write(const cv::Mat& frame) {
    if (!isRunning || frame.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(outputQueueMutex);
        
        // Keep queue size limited
        while (outputFrameQueue.size() >= static_cast<size_t>(params.queueSize)) {
            outputFrameQueue.pop();
        }
        
        outputFrameQueue.push(frame.clone());
        outputQueueCondition.notify_one();
    }
}

void CamCapInterpipe::inputLoop() {
    // This is handled by the GStreamer callback
    // Just keep the thread alive
    while (!terminate) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void CamCapInterpipe::outputLoop() {
    while (!terminate) {
        cv::Mat frame;
        
        {
            std::unique_lock<std::mutex> lock(outputQueueMutex);
            
            while (outputFrameQueue.empty() && !terminate) {
                outputQueueCondition.wait_for(lock, std::chrono::milliseconds(10));
            }
            
            if (terminate) {
                break;
            }
            
            if (!outputFrameQueue.empty()) {
                frame = outputFrameQueue.front();
                outputFrameQueue.pop();
            }
        }
        
        if (!frame.empty() && appSrc) {
            // Convert cv::Mat to GstBuffer
            size_t dataSize = frame.total() * frame.elemSize();
            GstBuffer* buffer = gst_buffer_new_allocate(nullptr, dataSize, nullptr);
            
            GstMapInfo mapInfo;
            gst_buffer_map(buffer, &mapInfo, GST_MAP_WRITE);
            memcpy(mapInfo.data, frame.data, dataSize);
            gst_buffer_unmap(buffer, &mapInfo);
            
            // Set timestamp
            static GstClockTime timestamp = 0;
            GST_BUFFER_PTS(buffer) = timestamp;
            GST_BUFFER_DURATION(buffer) = GST_SECOND / params.fps;
            timestamp += GST_BUFFER_DURATION(buffer);
            
            // Push buffer
            GstFlowReturn ret = gst_app_src_push_buffer(appSrc, buffer);
            
            if (ret != GST_FLOW_OK && params.logging) {
                std::cerr << "[CamCapInterpipe] Failed to push buffer: " << ret << std::endl;
            }
        }
    }
}

GstFlowReturn CamCapInterpipe::newSampleCallback(GstAppSink* sink, gpointer userData) {
    CamCapInterpipe* self = static_cast<CamCapInterpipe*>(userData);
    
    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) {
        return GST_FLOW_ERROR;
    }
    
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    
    // Get frame info from caps
    GstStructure* structure = gst_caps_get_structure(caps, 0);
    int width, height;
    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);
    
    // Map buffer
    GstMapInfo mapInfo;
    gst_buffer_map(buffer, &mapInfo, GST_MAP_READ);
    
    // Create cv::Mat
    cv::Mat frame(height, width, CV_8UC3, mapInfo.data);
    
    // Add to queue
    {
        std::lock_guard<std::mutex> lock(self->inputQueueMutex);
        
        // Keep queue size limited
        while (self->inputFrameQueue.size() >= static_cast<size_t>(self->params.queueSize)) {
            self->inputFrameQueue.pop();
        }
        
        self->inputFrameQueue.push(frame.clone());
        self->inputQueueCondition.notify_one();
    }
    
    gst_buffer_unmap(buffer, &mapInfo);
    gst_sample_unref(sample);
    
    return GST_FLOW_OK;
}

gboolean CamCapInterpipe::busCallback(GstBus* bus, GstMessage* message, gpointer userData) {
    CamCapInterpipe* self = static_cast<CamCapInterpipe*>(userData);
    
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR: {
            GError* error;
            gchar* debug;
            gst_message_parse_error(message, &error, &debug);
            
            if (self->params.logging) {
                std::cerr << "[CamCapInterpipe] Error: " << error->message << std::endl;
                if (debug) {
                    std::cerr << "[CamCapInterpipe] Debug: " << debug << std::endl;
                }
            }
            
            g_error_free(error);
            g_free(debug);
            break;
        }
        case GST_MESSAGE_WARNING: {
            GError* warning;
            gchar* debug;
            gst_message_parse_warning(message, &warning, &debug);
            
            if (self->params.logging) {
                std::cerr << "[CamCapInterpipe] Warning: " << warning->message << std::endl;
                if (debug) {
                    std::cerr << "[CamCapInterpipe] Debug: " << debug << std::endl;
                }
            }
            
            g_error_free(warning);
            g_free(debug);
            break;
        }
        default:
            break;
    }
    
    return TRUE;
}

bool CamCapInterpipe::isHealthy() const {
    return initialized && isRunning && 
           inputPipeline && outputPipeline && 
           appSink && appSrc;
}

} // namespace vs
