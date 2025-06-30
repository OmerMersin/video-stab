#include "video/DeepStreamTracker.h"
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <iomanip>

namespace vs {

DeepStreamTracker::DeepStreamTracker(const Parameters& params) 
    : params_(params), 
      lastReportTime_(std::chrono::high_resolution_clock::now()) {
    
    // Create directory for saved images if enabled
    if (params_.saveDetectionImages) {
        createDirIfNotExists(params_.saveImagePath);
    }
}

DeepStreamTracker::~DeepStreamTracker() {
    release();
}

bool DeepStreamTracker::initialize() {
    if (initialized_) {
        return true;
    }
    
    // Initialize GStreamer once
    static bool gstInitialized = false;
    if (!gstInitialized) {
        gst_init(nullptr, nullptr);
        gstInitialized = true;
    }
    
    // Create pipeline
    if (!createPipeline()) {
        return false;
    }
    
    // Start processing thread
    stopProcessing_ = false;
    processingThread_ = std::thread(&DeepStreamTracker::processingLoop, this);
    
    initialized_ = true;
    std::cout << "DeepStream tracker initialized successfully" << std::endl;
    
    return true;
}

void DeepStreamTracker::release() {
    if (initialized_) {
        // Stop processing thread
        stopProcessing_ = true;
        frameCondition_.notify_all();
        if (processingThread_.joinable()) {
            processingThread_.join();
        }
        
        // Stop pipeline
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
        
        initialized_ = false;
    }
}

std::vector<DeepStreamTracker::Detection> DeepStreamTracker::processFrame(const cv::Mat& frame) {
    auto startTime = std::chrono::high_resolution_clock::now();
    //----------------------------------------------------------
   // 1. skip completely empty frames (size()==0 or just 0×0)
   //----------------------------------------------------------
   if (frame.empty() || frame.cols == 0 || frame.rows == 0)
       return {};                    // nothing to track this time

    if (!initialized_ && !initialize()) {
        return {};
    }
    
    // Process the frame
    cv::Mat resizedFrame;
    if (frame.cols != params_.processingWidth || frame.rows != params_.processingHeight) {
        cv::resize(frame, resizedFrame, cv::Size(params_.processingWidth, params_.processingHeight));
    } else {
        resizedFrame = frame.clone();
    }
    
    cv::Mat rgbaFrame;
    cv::cvtColor(resizedFrame, rgbaFrame, cv::COLOR_BGR2RGBA);
    
    // Push frame to queue
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (params_.enableLowLatency) {
            // Keep only the most recent frame
            while (!inputFrameQueue_.empty()) {
                inputFrameQueue_.pop();
            }
        }
        inputFrameQueue_.push(rgbaFrame);
    }
    
    // Notify processing thread
    frameCondition_.notify_one();
    
    // Get detections
    std::vector<Detection> currentDetections;
    {
        std::lock_guard<std::mutex> lock(detectionsMutex_);
        currentDetections = latestDetections_;
    }
    
    // Update performance metrics
    auto endTime = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    totalProcessingTime_.store(totalProcessingTime_.load() + processingTime);
            frameCount_++;
    
    // Report performance every 30 frames
    if (frameCount_ % 30 == 0) {
        reportPerformance();
    }
    
    return currentDetections;
}

//--------------------------------------------------------------
//  Draw detections
//  – when a new (selX , selY) pair arrives, find the trackId of the
//    first bbox that contains that point and remember it.
//  – afterwards draw *only* that ID until the next coordinates arrive.
//--------------------------------------------------------------
cv::Mat DeepStreamTracker::drawDetections(const cv::Mat& frame,
                                         const std::vector<Detection>& detections,
                                         int selX,
                                         int selY)
{
    if (frame.empty())           // nothing to do
        return frame;

    static int selectedId = -1;  // remembered between calls
    static bool selectionMode = false; // Track if we're in selection mode

    /* ------ 1. update 'selectedId' if fresh coordinates came in ------ */
    if (selX >= 0 && selY >= 0)          // (-1,-1) means "no new point"
    {
        // Enter selection mode when new coordinates are received
        selectionMode = true;
        
        // Only reset selectedId if we successfully find a new one
        int newSelectedId = -1;
        
        if (!detections.empty())
        {
            double sx = static_cast<double>(frame.cols) / params_.processingWidth;
            double sy = static_cast<double>(frame.rows) / params_.processingHeight;
            
            std::cout << "Looking for object at (" << selX << "," << selY 
                      << ") in " << detections.size() << " detections" << std::endl;

            for (const auto& d : detections)
            {
                std::cout << "  Checking trackId " << d.trackId 
                          << " at " << d.bbox.x << "," << d.bbox.y 
                          << " size " << d.bbox.width << "x" << d.bbox.height << std::endl;

                // Use the helper function instead of creating rectangle and checking manually
                if (isPointInDetection(d, selX, selY, sx, sy))
                {
                    newSelectedId = d.trackId;
                    std::cout << "  Found match: trackId " << newSelectedId << std::endl;
                    break;
                }
            }
        }
        
        // Only update if we found a valid object
        if (newSelectedId != -1) {
            selectedId = newSelectedId;
            std::cout << "Selected object with ID: " << selectedId << std::endl;
        } else {
            std::cout << "No object found at coordinates (" << selX << "," << selY 
                      << "), keeping previous selection: " << selectedId << std::endl;
        }
    }
    
    // Check if we should exit selection mode (press Esc key or click outside all objects)
    if (selX == 9999 && selY == 9999) {  // Special signal to exit selection mode
        selectionMode = false;
        selectedId = -1;
        std::cout << "Exiting selection mode, showing all objects" << std::endl;
    }
    /* ----------------------------------------------------------------- */

    cv::Mat out = frame.clone();
    if (detections.empty())
        return out;                              // nothing to draw

    /* ------ 2. colour palette (8 distinct colours) ------ */
    static const cv::Scalar palette[] = {
        {  0,  0,255}, {  0,255,  0}, {255, 0,  0}, {  0,255,255},
        {255, 0,255}, {255,255,  0}, {255,255,255}, {128,128,128}
    };
    constexpr int N_COLOURS = int(sizeof(palette)/sizeof(palette[0]));

    double sx = static_cast<double>(out.cols) / params_.processingWidth;
    double sy = static_cast<double>(out.rows) / params_.processingHeight;

    /* ------ 3. draw detections based on selection state ------ */
    bool drawAll = !selectionMode || selectedId == -1;
    
    // Check if the selected object is still present
    bool selectedObjectFound = false;
    if (!drawAll) {
        for (const auto& d : detections) {
            if (d.trackId == selectedId) {
                selectedObjectFound = true;
                break;
            }
        }
        
        // If selected object is not found, decide whether to draw all or none
        if (!selectedObjectFound) {
            std::cout << "Selected object ID " << selectedId << " is no longer visible" << std::endl;
            // Option 1: Draw all objects when selected one disappears
            drawAll = true;
            // Option 2: Keep looking for the selected object (uncomment to use this)
            // return out; // Return empty frame with no detections drawn
        }
    }
    
    for (const auto& d : detections)
    {
        // Skip if we're not drawing all and this isn't the selected ID
        if (!drawAll && d.trackId != selectedId)
            continue;

        cv::Rect box(lround(d.bbox.x      * sx),
                     lround(d.bbox.y      * sy),
                     lround(d.bbox.width  * sx),
                     lround(d.bbox.height * sy));
        box &= cv::Rect(0, 0, out.cols, out.rows);   // clip to image

        cv::Scalar col = palette[d.classId % N_COLOURS];
        cv::rectangle(out, box, col, 2);

        std::ostringstream txt;
        txt << d.label << "  ID:" << d.trackId
            << "  " << int(d.confidence * 100) << '%';

        int baseLine = 0;
        cv::Size tsize = cv::getTextSize(txt.str(),
                                         cv::FONT_HERSHEY_SIMPLEX,
                                         0.5, 1, &baseLine);

        cv::rectangle(out,
                      {box.x, box.y - tsize.height - 6},
                      {box.x + tsize.width, box.y},
                      col, cv::FILLED);

        cv::putText(out, txt.str(),
                    {box.x, box.y - 3},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    CV_RGB(0,0,0), 1);
                    
        // If drawing just the selected ID, we can break after the first match
        if (!drawAll)
            break;
    }

    /* ------ 4. optional FPS overlay ----------------------- */
    double avgFps = (frameCount_ > 0)
                  ? (1000.0 * frameCount_ / totalProcessingTime_)
                  : 0.0;
    std::ostringstream fps;
    fps << "FPS: " << std::fixed << std::setprecision(1) << avgFps;
    cv::putText(out, fps.str(), {10,30},
                cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
                
    // Add selection mode indicator
    if (selectionMode && selectedId != -1) {
        std::ostringstream mode;
        mode << "TRACKING ID: " << selectedId;
        cv::putText(out, mode.str(), {10,60},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,255}, 2);
    }

    return out;
}

// Helper function to check if a point is inside a detection
bool DeepStreamTracker::isPointInDetection(const Detection& detection, 
                                          int x, int y, 
                                          double scaleX, double scaleY) const {
    cv::Rect r(static_cast<int>(detection.bbox.x * scaleX),
              static_cast<int>(detection.bbox.y * scaleY),
              static_cast<int>(detection.bbox.width * scaleX),
              static_cast<int>(detection.bbox.height * scaleY));
    
    return r.contains(cv::Point(x, y));
}

int DeepStreamTracker::pickIdAt(int x, int y) const
{
    std::lock_guard<std::mutex> lock(detectionsMutex_);

    double sx = static_cast<double>(params_.processingWidth);
    double sy = static_cast<double>(params_.processingHeight);

    for (const auto& d : latestDetections_)
    {
        cv::Rect r(static_cast<int>(d.bbox.x),
                   static_cast<int>(d.bbox.y),
                   static_cast<int>(d.bbox.width),
                   static_cast<int>(d.bbox.height));

        if (r.contains(cv::Point(static_cast<int>(x / sx), static_cast<int>(y / sy))))
            return d.trackId;            // first hit wins
    }
    return -1;
}


std::string DeepStreamTracker::getLastError() const {
    return lastErrorMessage_;
}

bool DeepStreamTracker::createPipeline() {
    // Create DeepStream pipeline
    pipeline_ = gst_pipeline_new("tracker-pipeline");
    if (!pipeline_) {
        setLastError("Failed to create pipeline");
        return false;
    }
    
    // Create elements
    GstElement* appsrc = gst_element_factory_make("appsrc", "app-source");
    GstElement* videoconvert = gst_element_factory_make("videoconvert", "convert1");
    GstElement* nvvideoconvert = gst_element_factory_make("nvvideoconvert", "nvconvert");
    GstElement* capsfilter = gst_element_factory_make("capsfilter", "filter");
    GstElement* streammux = gst_element_factory_make("nvstreammux", "stream-mux");
    GstElement* pgie = gst_element_factory_make("nvinfer", "primary-inference");
    GstElement* tracker = gst_element_factory_make("nvtracker", "tracker");
    GstElement* nvvideoconvert2 = gst_element_factory_make("nvvideoconvert", "nvconvert2");
    GstElement* nvosd = gst_element_factory_make("nvdsosd", "nvosd");
    GstElement* videoconvert2 = gst_element_factory_make("videoconvert", "convert2");
    GstElement* appsink = gst_element_factory_make("appsink", "app-sink");
    
    if (!appsrc || !videoconvert || !nvvideoconvert || !capsfilter || !streammux || !pgie || 
        !tracker || !nvvideoconvert2 || !nvosd || !videoconvert2 || !appsink) {
        setLastError("Failed to create one or more elements");
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return false;
    }
    
    // Configure AppSrc
    g_object_set(G_OBJECT(appsrc),
                "stream-type", 0, // GST_APP_STREAM_TYPE_STREAM
                "format", GST_FORMAT_TIME,
                "is-live", TRUE,
                "do-timestamp", TRUE,
                NULL);
                
    g_object_set(G_OBJECT(nvvideoconvert), "nvbuf-memory-type", 0, NULL);
    
    // Configure AppSrc caps - use BGR since that's what OpenCV provides
    GstCaps* caps = gst_caps_new_simple("video/x-raw",
                                      "format", G_TYPE_STRING, "RGBA",
                                      "width", G_TYPE_INT, params_.processingWidth,
                                      "height", G_TYPE_INT, params_.processingHeight,
                                      "framerate", GST_TYPE_FRACTION, 30, 1,
                                      NULL);
    gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
    gst_caps_unref(caps);
    
    // Configure a capsfilter to ensure NVMM memory for nvvideoconvert output
    GstCaps* nvmmCaps = gst_caps_new_simple("video/x-raw(memory:NVMM)",
                                         "format", G_TYPE_STRING, "NV12",
                                         "width", G_TYPE_INT, params_.processingWidth,
                                         "height", G_TYPE_INT, params_.processingHeight,
                                         NULL);
    g_object_set(G_OBJECT(capsfilter), "caps", nvmmCaps, NULL);
    gst_caps_unref(nvmmCaps);
    
    // Configure StreamMux
    g_object_set(G_OBJECT(streammux),
                "batch-size", params_.batchSize,
                "width", params_.processingWidth,
                "height", params_.processingHeight,
                "batched-push-timeout", 40000,
                "live-source", 1,
                NULL);
    
    // Configure primary inference engine with ResNet-18 model
    g_object_set(G_OBJECT(pgie),
                "config-file-path", params_.modelConfigFile.c_str(),
                "model-engine-file", params_.modelEngine.c_str(),
                NULL);
    
    // Configure tracker
    g_object_set(G_OBJECT(tracker),
                "tracker-width", params_.processingWidth,
                "tracker-height", params_.processingHeight,
                "gpu-id", params_.gpuId,
                "ll-lib-file", "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so",
                "ll-config-file", params_.trackerConfigFile.c_str(),
                "compute-hw", 1, // Use GPU
                NULL);
    
    // Configure AppSink
    g_object_set(G_OBJECT(appsink),
                "emit-signals", TRUE,
                "max-buffers", 1,
                "drop", TRUE,
                "sync", FALSE,
                NULL);
    
    // Add elements to pipeline
    gst_bin_add_many(GST_BIN(pipeline_), 
                    appsrc, videoconvert, nvvideoconvert, capsfilter, streammux,
                    pgie, tracker, nvvideoconvert2, nvosd, 
                    videoconvert2, appsink, NULL);
    
    // Link elements up to nvvideoconvert
    if (!gst_element_link_many(appsrc, videoconvert, nvvideoconvert, capsfilter, NULL)) {
        setLastError("Failed to link appsrc->capsfilter");
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return false;
    }
    
    // Link capsfilter to streammux (request pad)
    GstPad* srcpad = gst_element_get_static_pad(capsfilter, "src");
    GstPad* sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
    
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        setLastError("Failed to link capsfilter to streammux");
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return false;
    }
    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
    
    // Link streammux to appsink
    if (!gst_element_link_many(streammux, pgie, tracker, nvvideoconvert2, nvosd, 
                             videoconvert2, appsink, NULL)) {
        setLastError("Failed to link streammux->appsink");
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return false;
    }
    
    // Add probe to extract metadata from NVOSD sink pad
    GstPad* osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, 
                    metadataProbeCallback, this, NULL);
    gst_object_unref(osd_sink_pad);
    
    // Set pipeline to PLAYING state
    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        setLastError("Failed to set pipeline to playing state");
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return false;
    }
    
    return true;
}

void DeepStreamTracker::processingLoop() {
    GstElement* appsrc = gst_bin_get_by_name(GST_BIN(pipeline_), "app-source");
    
    if (!appsrc) {
        setLastError("Failed to get appsrc element");
        return;
    }
    
    while (!stopProcessing_) {
        // Wait for a frame
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            frameCondition_.wait_for(lock, std::chrono::milliseconds(100), 
                                   [this] { return !inputFrameQueue_.empty() || stopProcessing_; });
            
            if (stopProcessing_) break;
            if (inputFrameQueue_.empty()) continue;
            
            frame = inputFrameQueue_.front();
            inputFrameQueue_.pop();
        }
        
        // Convert frame to GstBuffer and push to appsrc
        GstBuffer* buffer = matToGstBuffer(frame);
        if (buffer) {
            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
            if (ret != GST_FLOW_OK) {
                std::cerr << "Failed to push buffer to appsrc: " << ret << std::endl;
            }
        }
    }
    
    gst_object_unref(appsrc);
}

GstBuffer* DeepStreamTracker::matToGstBuffer(const cv::Mat& frame) {
    if (frame.empty()) {
        return nullptr;
    }
    
    // Allocate buffer for BGR data
    gsize size = frame.total() * frame.elemSize();
    GstBuffer* buffer = gst_buffer_new_allocate(NULL, size, NULL);
    if (!buffer) {
        setLastError("Failed to allocate GstBuffer");
        return nullptr;
    }
    
    // Map buffer for writing
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        setLastError("Failed to map GstBuffer");
        gst_buffer_unref(buffer);
        return nullptr;
    }
    
    // Copy frame data to buffer
    memcpy(map.data, frame.data, size);
    gst_buffer_unmap(buffer, &map);
    
    return buffer;
}

GstPadProbeReturn DeepStreamTracker::metadataProbeCallback(GstPad* pad, GstPadProbeInfo* info, gpointer userData) {
    DeepStreamTracker* self = static_cast<DeepStreamTracker*>(userData);
    GstBuffer* buffer = (GstBuffer*)info->data;
    
    // Get metadata from buffer
    NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batchMeta) {
        return GST_PAD_PROBE_OK;
    }
    
    // Process each frame's metadata
    for (NvDsMetaList* frameMetaList = batchMeta->frame_meta_list; frameMetaList != NULL; frameMetaList = frameMetaList->next) {
        NvDsFrameMeta* frameMeta = (NvDsFrameMeta*)(frameMetaList->data);
        self->processMetadata(frameMeta);
    }
    
    return GST_PAD_PROBE_OK;
}

void DeepStreamTracker::processMetadata(NvDsFrameMeta* frameMeta) {
    std::vector<Detection> detections;
    
    // Process object metadata
    for (NvDsMetaList* objMetaList = frameMeta->obj_meta_list; objMetaList != NULL; objMetaList = objMetaList->next) {
        NvDsObjectMeta* objMeta = (NvDsObjectMeta*)(objMetaList->data);
        
        // Skip objects with confidence below threshold
        if (objMeta->confidence < params_.confidenceThreshold) {
            continue;
        }
        
        // Extract detection info
        Detection det;
        det.classId = objMeta->class_id;
        det.confidence = objMeta->confidence;
        det.bbox = cv::Rect(
            static_cast<int>(objMeta->rect_params.left),
            static_cast<int>(objMeta->rect_params.top),
            static_cast<int>(objMeta->rect_params.width),
            static_cast<int>(objMeta->rect_params.height)
        );
        det.trackId = objMeta->object_id;
        det.label = objMeta->obj_label ? std::string(objMeta->obj_label) : 
                   "Class " + std::to_string(objMeta->class_id);
        
        detections.push_back(det);
    }
    
    // Save detected objects count
    std::cout << "Detected " << detections.size() << " objects" << std::endl;
    
    // Update latest detections
    {
        std::lock_guard<std::mutex> lock(detectionsMutex_);
        latestDetections_ = detections;
    }
}

void DeepStreamTracker::reportPerformance() {
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - lastReportTime_).count();
    lastReportTime_ = now;
    
    double avgProcessTime = totalProcessingTime_ / frameCount_;
    double fps = frameCount_ / elapsed;
    
    std::cout << "Processing Time: " << std::fixed << std::setprecision(4) 
              << avgProcessTime << " ms | FPS: " << fps << std::endl;
}

void DeepStreamTracker::setLastError(const std::string& error) {
    lastErrorMessage_ = error;
    std::cerr << "DeepStreamTracker Error: " << error << std::endl;
}

bool DeepStreamTracker::createDirIfNotExists(const std::string& path) {
    try {
        if (!std::filesystem::exists(path)) {
            return std::filesystem::create_directories(path);
        }
        return true;
    } catch (const std::exception& e) {
        setLastError("Failed to create directory: " + std::string(e.what()));
        return false;
    }
}

} // namespace vs
