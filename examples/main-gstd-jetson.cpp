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

// NVIDIA Jetson Multimedia API includes
#include "NvVideoEncoder.h"
#include "NvVideoDecoder.h"
#include "NvUtils.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include <linux/videodev2.h>
#include <malloc.h>
#include <pthread.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#define MAX_PLANES 3

// Global variable for signal handling
volatile sig_atomic_t stopRequested = 0;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down gracefully..." << std::endl;
    stopRequested = 1;
}

// Forward declaration
class OptimizedPipelineManager;

// Hardware encoder context for low-level NVIDIA API
struct HardwareEncoderContext {
    NvVideoEncoder* encoder;
    EGLDisplay eglDisplay;
    int width, height;
    double fps;
    int bitrate;
    bool initialized;
    bool got_error;
    bool got_eos;
    std::ofstream* outputStream;
    pthread_t encoderThread;
    void* pipelineManager;  // Pointer to parent pipeline manager
    
    HardwareEncoderContext() : encoder(nullptr), eglDisplay(EGL_NO_DISPLAY), 
                              width(0), height(0), fps(30.0), bitrate(4000000),
                              initialized(false), got_error(false), got_eos(false),
                              outputStream(nullptr), pipelineManager(nullptr) {}
    
    ~HardwareEncoderContext() {
        cleanup();
    }
    
    void cleanup() {
        if (encoder) {
            encoder->capture_plane.waitForDQThread(2000);
            delete encoder;
            encoder = nullptr;
        }
        if (outputStream) {
            outputStream->close();
            delete outputStream;
            outputStream = nullptr;
        }
        if (eglDisplay != EGL_NO_DISPLAY) {
            eglTerminate(eglDisplay);
            eglReleaseThread();
            eglDisplay = EGL_NO_DISPLAY;
        }
    }
};

// Forward declaration of callback function
static bool encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer *buffer,
                                              NvBuffer *shared_buffer, void *arg);

// Function to convert OpenCV Mat to NvBuffer
static int mat_to_nvbuffer(const cv::Mat& mat, NvBuffer* buffer) {
    if (mat.empty() || !buffer) return -1;
    
    // Ensure the buffer has enough space for BGR24 format
    size_t required_size = mat.rows * mat.cols * 3; // BGR24 format
    if (buffer->planes[0].length < required_size) {
        std::cerr << "Buffer too small for OpenCV Mat: " << buffer->planes[0].length << " < " << required_size << std::endl;
        return -1;
    }
    
    // Convert to BGR format if needed
    cv::Mat bgr_mat;
    if (mat.channels() == 1) {
        cv::cvtColor(mat, bgr_mat, cv::COLOR_GRAY2BGR);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, bgr_mat, cv::COLOR_BGRA2BGR);
    } else if (mat.channels() == 3) {
        bgr_mat = mat;
    } else {
        std::cerr << "Unsupported number of channels: " << mat.channels() << std::endl;
        return -1;
    }
    
    // Copy BGR data from OpenCV Mat to NvBuffer
    if (bgr_mat.isContinuous()) {
        memcpy(buffer->planes[0].data, bgr_mat.data, required_size);
    } else {
        // Handle non-continuous Mat
        uint8_t* dst = (uint8_t*)buffer->planes[0].data;
        for (int i = 0; i < bgr_mat.rows; i++) {
            memcpy(dst + i * bgr_mat.cols * 3, bgr_mat.ptr(i), bgr_mat.cols * 3);
        }
    }
    
    buffer->planes[0].bytesused = required_size;
    
    return 0;
}

// Function to sync buffer for device access
static int sync_buffer_for_device(NvBuffer *buffer) {
    NvBufSurface *nvbuf_surf = nullptr;
    int ret = NvBufSurfaceFromFd(buffer->planes[0].fd, (void**)(&nvbuf_surf));
    if (ret < 0) {
        std::cerr << "NvBufSurfaceFromFd failed!" << std::endl;
        return -1;
    }
    return NvBufSurfaceSyncForDevice(nvbuf_surf, -1, -1);
}

// Optimized Pipeline Manager with Hardware Encoding
// Uses passthrough for direct H.265 forwarding and hardware encoder for processing
class OptimizedPipelineManager {
private:
    std::string sourceAddress;
    std::string outputAddress;
    int frameWidth, frameHeight;
    double fps;
    int bitrate;
    bool pipelinesInitialized;
    
    // Passthrough pipeline (GStreamer for direct forwarding)
    GstElement* passthroughPipeline;
    
    // Hardware encoder for processing pipeline
    HardwareEncoderContext hwEncoderCtx;
    
    // RTSP output pipeline for hardware encoded stream
    GstElement* rtspOutputPipeline;
    GstAppSrc* rtspAppsrc;
    
    bool streamInitialized;
    uint64_t frameCounter;
    bool isCurrentlyPassthrough;
    bool usingSoftwareEncoder;  // Flag to track if we're using software encoder fallback
    std::string softwareEncoderType;  // Track which software encoder (x264 or x265)
    
    // GStreamer callback for bus messages
    static gboolean busCallback(GstBus* bus, GstMessage* msg, gpointer data) {
        OptimizedPipelineManager* manager = static_cast<OptimizedPipelineManager*>(data);
        
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
                std::cout << "Pipeline state changed from " << gst_element_state_get_name(old_state) 
                          << " to " << gst_element_state_get_name(new_state) << std::endl;
                break;
            }
            default:
                break;
        }
        return TRUE;
    }
    
    bool initializeHardwareEncoder() {
        std::cout << "Attempting to initialize hardware encoder..." << std::endl;
        
        // Initialize EGL display
        hwEncoderCtx.eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (hwEncoderCtx.eglDisplay == EGL_NO_DISPLAY) {
            std::cerr << "Could not get EGL display connection" << std::endl;
            return false;
        }
        
        if (!eglInitialize(hwEncoderCtx.eglDisplay, NULL, NULL)) {
            std::cerr << "Failed to initialize EGL display" << std::endl;
            return false;
        }
        
        // Create hardware encoder
        std::cout << "Creating hardware encoder instance..." << std::endl;
        hwEncoderCtx.encoder = NvVideoEncoder::createVideoEncoder("enc0");
        if (!hwEncoderCtx.encoder) {
            std::cerr << "Could not create hardware encoder - this could be due to:" << std::endl;
            std::cerr << "  1. Hardware encoder is busy or in use by another process" << std::endl;
            std::cerr << "  2. Driver issues with NVIDIA multimedia API" << std::endl;
            std::cerr << "  3. Insufficient permissions or resources" << std::endl;
            std::cerr << "  4. Hardware encoder not available on this device" << std::endl;
            return false;
        }
        
        std::cout << "Hardware encoder instance created successfully" << std::endl;
        
        // Set encoder parameters
        hwEncoderCtx.width = frameWidth;
        hwEncoderCtx.height = frameHeight;
        hwEncoderCtx.fps = fps;
        hwEncoderCtx.bitrate = bitrate;
        
        // Configure encoder format (H.264 for compatibility)
        int ret = hwEncoderCtx.encoder->setCapturePlaneFormat(V4L2_PIX_FMT_H264, 
                                                              frameWidth, frameHeight, 
                                                              2 * 1024 * 1024);
        if (ret < 0) {
            std::cerr << "Could not set capture plane format" << std::endl;
            return false;
        }
        
        // Set output plane format (BGR input from OpenCV)
        ret = hwEncoderCtx.encoder->setOutputPlaneFormat(V4L2_PIX_FMT_BGR24, 
                                                         frameWidth, frameHeight);
        if (ret < 0) {
            std::cerr << "Could not set output plane format" << std::endl;
            return false;
        }
        
        // Set bitrate
        ret = hwEncoderCtx.encoder->setBitrate(bitrate);
        if (ret < 0) {
            std::cerr << "Could not set bitrate" << std::endl;
            return false;
        }
        
        // Set profile and level
        ret = hwEncoderCtx.encoder->setProfile(V4L2_MPEG_VIDEO_H264_PROFILE_HIGH);
        if (ret < 0) {
            std::cerr << "Could not set encoder profile" << std::endl;
            return false;
        }
        
        ret = hwEncoderCtx.encoder->setLevel(V4L2_MPEG_VIDEO_H264_LEVEL_5_0);
        if (ret < 0) {
            std::cerr << "Could not set encoder level" << std::endl;
            return false;
        }
        
        // Set framerate
        ret = hwEncoderCtx.encoder->setFrameRate(static_cast<uint32_t>(fps), 1);
        if (ret < 0) {
            std::cerr << "Could not set framerate" << std::endl;
            return false;
        }
        
        // Setup encoder planes
        ret = hwEncoderCtx.encoder->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
        if (ret < 0) {
            std::cerr << "Could not setup output plane" << std::endl;
            return false;
        }
        
        ret = hwEncoderCtx.encoder->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 6, true, false);
        if (ret < 0) {
            std::cerr << "Could not setup capture plane" << std::endl;
            return false;
        }
        
        // Start encoder streaming
        ret = hwEncoderCtx.encoder->output_plane.setStreamStatus(true);
        if (ret < 0) {
            std::cerr << "Error in output plane streamon" << std::endl;
            return false;
        }
        
        ret = hwEncoderCtx.encoder->capture_plane.setStreamStatus(true);
        if (ret < 0) {
            std::cerr << "Error in capture plane streamon" << std::endl;
            return false;
        }
        
        // Set callback for capture plane
        hwEncoderCtx.encoder->capture_plane.setDQThreadCallback(encoder_capture_plane_dq_callback);
        hwEncoderCtx.pipelineManager = this;  // Set parent pipeline manager
        hwEncoderCtx.encoder->capture_plane.startDQThread(&hwEncoderCtx);
        
        // Enqueue empty capture plane buffers
        for (uint32_t i = 0; i < hwEncoderCtx.encoder->capture_plane.getNumBuffers(); i++) {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];
            
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
            
            v4l2_buf.index = i;
            v4l2_buf.m.planes = planes;
            
            ret = hwEncoderCtx.encoder->capture_plane.qBuffer(v4l2_buf, NULL);
            if (ret < 0) {
                std::cerr << "Error while queueing buffer at capture plane" << std::endl;
                return false;
            }
        }
        
        hwEncoderCtx.initialized = true;
        std::cout << "Hardware encoder initialized successfully" << std::endl;
        return true;
    }
    
    bool initializeRTSPOutput(bool useSoftwareEncoder = false) {
        // Create RTSP output pipeline 
        std::string rtspOutputPipelineStr;
        
        if (useSoftwareEncoder) {
            // Use software encoder for dummy/invalid streams
            // Try x265 first (H.265), fallback to x264 (H.264) if x265 is not available
            rtspOutputPipelineStr = 
                "appsrc name=src is-live=true format=time block=false max-latency=0 "
                "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) + 
                ",height=" + std::to_string(frameHeight) + ",framerate=" + std::to_string(static_cast<int>(fps)) + "/1 ! "
                "videoconvert ! "
                "x265enc tune=zerolatency speed-preset=ultrafast bitrate=" + std::to_string(bitrate/1000) + " ! "
                "h265parse ! "
                "rtspclientsink location=" + outputAddress + " protocols=tcp latency=0";
            std::cout << "Using SOFTWARE encoder (x265/H.265) for RTSP output (fallback mode)" << std::endl;
            
            // Test if x265 pipeline can be created
            std::cout << "Testing x265 software encoder pipeline: " << rtspOutputPipelineStr << std::endl;
            GstElement* testPipeline = gst_parse_launch(rtspOutputPipelineStr.c_str(), nullptr);
            if (!testPipeline) {
                std::cerr << "x265 encoder not available, falling back to x264..." << std::endl;
                // Fallback to x264
                rtspOutputPipelineStr = 
                    "appsrc name=src is-live=true format=time block=false max-latency=0 "
                    "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) + 
                    ",height=" + std::to_string(frameHeight) + ",framerate=" + std::to_string(static_cast<int>(fps)) + "/1 ! "
                    "videoconvert ! "
                    "x264enc tune=zerolatency speed-preset=ultrafast bitrate=" + std::to_string(bitrate/1000) + " ! "
                    "h264parse ! "
                    "rtspclientsink location=" + outputAddress + " protocols=tcp latency=0";
                std::cout << "Using SOFTWARE encoder (x264/H.264) for RTSP output (fallback mode)" << std::endl;
                softwareEncoderType = "x264";
            } else {
                gst_object_unref(testPipeline);
                std::cout << "x265 software encoder available and will be used" << std::endl;
                softwareEncoderType = "x265";
            }
        } else {
            // Use hardware encoder (H.264 stream from hardware encoder)
            rtspOutputPipelineStr = 
                "appsrc name=src is-live=true format=time block=false max-latency=0 "
                "caps=video/x-h264,stream-format=byte-stream,alignment=au,profile=high ! "
                "h264parse ! "
                "rtspclientsink location=" + outputAddress + " protocols=tcp latency=0";
            std::cout << "Using HARDWARE encoder for RTSP output" << std::endl;
        }
        
        std::cout << "Creating RTSP output pipeline: " << rtspOutputPipelineStr << std::endl;
        rtspOutputPipeline = gst_parse_launch(rtspOutputPipelineStr.c_str(), nullptr);
        if (!rtspOutputPipeline) {
            std::cerr << "Failed to create RTSP output pipeline" << std::endl;
            return false;
        }
        
        // Get appsrc from RTSP output pipeline
        rtspAppsrc = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(rtspOutputPipeline), "src"));
        if (!rtspAppsrc) {
            std::cerr << "Failed to get RTSP appsrc element" << std::endl;
            return false;
        }
        
        // Set appsrc properties
        g_object_set(rtspAppsrc,
            "is-live", TRUE,
            "block", FALSE,
            "format", GST_FORMAT_TIME,
            "max-latency", G_GINT64_CONSTANT(0),
            "do-timestamp", TRUE,
            NULL);
        
        // Set up bus message handling
        GstBus* rtspBus = gst_pipeline_get_bus(GST_PIPELINE(rtspOutputPipeline));
        gst_bus_add_watch(rtspBus, busCallback, this);
        gst_object_unref(rtspBus);
        
        return true;
    }
    
public:
    OptimizedPipelineManager(const std::string& source = "rtsp://192.168.144.119:554",
                           const std::string& output = "rtsp://192.168.144.150:8554/forwarded", 
                           int bitrate = 4000000) 
        : sourceAddress(source), outputAddress(output), bitrate(bitrate), pipelinesInitialized(false),
          passthroughPipeline(nullptr), rtspOutputPipeline(nullptr), rtspAppsrc(nullptr),
          streamInitialized(false), frameCounter(0), isCurrentlyPassthrough(true), usingSoftwareEncoder(false),
          softwareEncoderType("none") {
        
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
        bitrate = calculatedBitrate * 1000; // Convert to bps
        
        // Check if this is a dummy/invalid stream (common patterns)
        bool isDummyStream = false;

        // Create PASSTHROUGH pipeline (direct H.265 forwarding)
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
        
        // Test passthrough pipeline first to validate stream
        std::cout << "Testing passthrough pipeline..." << std::endl;
        GstStateChangeReturn ret = gst_element_set_state(passthroughPipeline, GST_STATE_PAUSED);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "WARNING: Passthrough pipeline failed to pause - stream may be invalid" << std::endl;
            isDummyStream = false;
        } else {
            // Wait for state change
            gst_element_get_state(passthroughPipeline, nullptr, nullptr, 3 * GST_SECOND);
        }
        
        // Set back to NULL state for now
        gst_element_set_state(passthroughPipeline, GST_STATE_NULL);
        
        // Initialize hardware encoder for processing mode (skip for dummy streams)
        if (!isDummyStream && !initializeHardwareEncoder()) {
            std::cerr << "Failed to initialize hardware encoder, falling back to software encoding" << std::endl;
            isDummyStream = true; // Treat as dummy stream for software fallback
        }
        
        // Set software encoder flag
        usingSoftwareEncoder = isDummyStream;
        
        // Initialize RTSP output pipeline (use software encoder for dummy streams)
        if (!initializeRTSPOutput(isDummyStream)) {
            std::cerr << "Failed to initialize RTSP output pipeline" << std::endl;
            return false;
        }
        
        // Set up bus message handling for passthrough pipeline
        GstBus* passthroughBus = gst_pipeline_get_bus(GST_PIPELINE(passthroughPipeline));
        gst_bus_add_watch(passthroughBus, busCallback, this);
        gst_object_unref(passthroughBus);
        
        // Start in passthrough mode by default
        std::cout << "Starting passthrough pipeline..." << std::endl;
        if (gst_element_set_state(passthroughPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start passthrough pipeline" << std::endl;
            return false;
        }
        
        // Wait for passthrough pipeline to start
        gst_element_get_state(passthroughPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        pipelinesInitialized = true;
        streamInitialized = true;
        frameCounter = 0;
        isCurrentlyPassthrough = true;
        
        std::cout << "Optimized pipeline system initialized successfully" << std::endl;
        std::cout << "Source: " << sourceAddress << std::endl;
        std::cout << "Output: " << outputAddress << std::endl;
        std::cout << "Bitrate: " << calculatedBitrate << " Kbps" << std::endl;
        std::cout << "Started in PASSTHROUGH mode (direct H.265 forwarding)" << std::endl;
        return true;
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
        
        // Stop RTSP output pipeline
        std::cout << "Stopping RTSP output pipeline..." << std::endl;
        gst_element_set_state(rtspOutputPipeline, GST_STATE_NULL);
        gst_element_get_state(rtspOutputPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        // Start passthrough pipeline
        std::cout << "Starting passthrough pipeline..." << std::endl;
        if (gst_element_set_state(passthroughPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start passthrough pipeline" << std::endl;
            return false;
        }
        
        // Wait for passthrough pipeline to start
        gst_element_get_state(passthroughPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        isCurrentlyPassthrough = true;
        
        std::cout << "Switched to PASSTHROUGH mode - direct H.265 forwarding" << std::endl;
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
        
        // Stop passthrough pipeline
        std::cout << "Stopping passthrough pipeline..." << std::endl;
        gst_element_set_state(passthroughPipeline, GST_STATE_NULL);
        gst_element_get_state(passthroughPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        // Start RTSP output pipeline
        std::cout << "Starting RTSP output pipeline..." << std::endl;
        if (gst_element_set_state(rtspOutputPipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start RTSP output pipeline" << std::endl;
            return false;
        }
        
        // Wait for RTSP output pipeline to start
        gst_element_get_state(rtspOutputPipeline, nullptr, nullptr, 5 * GST_SECOND);
        
        isCurrentlyPassthrough = false;
        
        std::cout << "Switched to PROCESSING mode - " << (usingSoftwareEncoder ? ("software " + softwareEncoderType) : "hardware") << " encoding" << std::endl;
        return true;
    }
    
    bool pushFrame(const cv::Mat& frame) {
        if (!pipelinesInitialized || isCurrentlyPassthrough) {
            return false; // No need to push frames in passthrough mode
        }
        
        // If using software encoder, push raw frame directly to GStreamer
        if (usingSoftwareEncoder) {
            return pushRawFrame(frame);
        }
        
        // Hardware encoder path
        if (!hwEncoderCtx.initialized || hwEncoderCtx.got_error) {
            std::cerr << "Hardware encoder not ready or in error state" << std::endl;
            return false;
        }
        
        // Get available output buffer from encoder
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;
        
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.m.planes = planes;
        
        // Dequeue buffer from encoder output plane
        if (hwEncoderCtx.encoder->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, 10) < 0) {
            std::cerr << "Error while dequeuing buffer from encoder output plane" << std::endl;
            return false;
        }
        
        // Convert OpenCV Mat to NvBuffer
        if (mat_to_nvbuffer(frame, buffer) < 0) {
            std::cerr << "Error converting OpenCV Mat to NvBuffer" << std::endl;
            return false;
        }
        
        // Sync buffer for device access
        if (sync_buffer_for_device(buffer) < 0) {
            std::cerr << "Error syncing buffer for device" << std::endl;
            return false;
        }
        
        // Queue buffer back to encoder
        if (hwEncoderCtx.encoder->output_plane.qBuffer(v4l2_buf, NULL) < 0) {
            std::cerr << "Error while queueing buffer to encoder output plane" << std::endl;
            return false;
        }
        
        frameCounter++;
        return true;
    }
    
    bool pushEncodedData(void* data, size_t size) {
        if (!rtspAppsrc || isCurrentlyPassthrough || size == 0) {
            return false;
        }
        
        // Create GStreamer buffer from encoded H.264 data
        GstBuffer *buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
        if (!buffer) {
            std::cerr << "Failed to allocate GStreamer buffer for encoded data" << std::endl;
            return false;
        }
        
        // Map buffer for writing
        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            std::cerr << "Failed to map GStreamer buffer for encoded data" << std::endl;
            gst_buffer_unref(buffer);
            return false;
        }
        
        // Copy encoded data to buffer
        memcpy(map.data, data, size);
        gst_buffer_unmap(buffer, &map);
        
        // Set timestamp for encoded data
        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frameCounter, GST_SECOND, fps);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, fps);
        
        // Push buffer to RTSP output pipeline
        GstFlowReturn flowRet = gst_app_src_push_buffer(rtspAppsrc, buffer);
        
        if (flowRet != GST_FLOW_OK) {
            std::cerr << "Failed to push encoded data to RTSP pipeline: " << gst_flow_get_name(flowRet) << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool pushRawFrame(const cv::Mat& frame) {
        if (!rtspAppsrc || isCurrentlyPassthrough || frame.empty()) {
            return false;
        }
        
        // Create GStreamer buffer from raw BGR data
        size_t frameSize = frame.rows * frame.cols * frame.channels();
        GstBuffer *buffer = gst_buffer_new_allocate(nullptr, frameSize, nullptr);
        if (!buffer) {
            std::cerr << "Failed to allocate GStreamer buffer for raw frame" << std::endl;
            return false;
        }
        
        // Map buffer for writing
        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            std::cerr << "Failed to map GStreamer buffer for raw frame" << std::endl;
            gst_buffer_unref(buffer);
            return false;
        }
        
        // Copy raw BGR data to buffer
        if (frame.isContinuous()) {
            memcpy(map.data, frame.data, frameSize);
        } else {
            // Handle non-continuous Mat
            uint8_t* dst = map.data;
            for (int i = 0; i < frame.rows; i++) {
                memcpy(dst + i * frame.cols * frame.channels(), frame.ptr(i), frame.cols * frame.channels());
            }
        }
        
        gst_buffer_unmap(buffer, &map);
        
        // Set timestamp for raw frame
        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frameCounter, GST_SECOND, fps);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, fps);
        
        // Push buffer to RTSP output pipeline
        GstFlowReturn flowRet = gst_app_src_push_buffer(rtspAppsrc, buffer);
        
        if (flowRet != GST_FLOW_OK) {
            std::cerr << "Failed to push raw frame to RTSP pipeline: " << gst_flow_get_name(flowRet) << std::endl;
            return false;
        }
        
        frameCounter++;
        return true;
    }
    
    void printStatus() {
        if (!streamInitialized) {
            std::cout << "Pipeline not initialized" << std::endl;
            return;
        }
        
        std::cout << "=== Pipeline Status ===" << std::endl;
        std::cout << "Mode: " << (isCurrentlyPassthrough ? "PASSTHROUGH" : "PROCESSING") << std::endl;
        std::cout << "Frames processed: " << frameCounter << std::endl;
        std::cout << "Resolution: " << frameWidth << "x" << frameHeight << std::endl;
        std::cout << "FPS: " << fps << std::endl;
        std::cout << "Bitrate: " << bitrate << " bps" << std::endl;
        if (!isCurrentlyPassthrough) {
            if (usingSoftwareEncoder) {
                std::cout << "Encoder: SOFTWARE (" << softwareEncoderType << ")" << std::endl;
            } else {
                std::cout << "Encoder: HARDWARE (" << (hwEncoderCtx.initialized ? "Ready" : "Not ready") << ")" << std::endl;
            }
        }
        std::cout << "====================" << std::endl;
    }
    
    void cleanup() {
        if (passthroughPipeline) {
            gst_element_set_state(passthroughPipeline, GST_STATE_NULL);
            gst_object_unref(passthroughPipeline);
            passthroughPipeline = nullptr;
        }
        
        if (rtspOutputPipeline) {
            gst_element_set_state(rtspOutputPipeline, GST_STATE_NULL);
            gst_object_unref(rtspOutputPipeline);
            rtspOutputPipeline = nullptr;
        }
        
        hwEncoderCtx.cleanup();
        
        streamInitialized = false;
        pipelinesInitialized = false;
        
        std::cout << "Pipeline cleanup completed" << std::endl;
    }
    
    ~OptimizedPipelineManager() {
        cleanup();
    }
};

// Hardware encoder callback function implementation
static bool encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer *buffer,
                                              NvBuffer *shared_buffer, void *arg) {
    HardwareEncoderContext *ctx = (HardwareEncoderContext *) arg;
    
    if (!v4l2_buf) {
        std::cerr << "Failed to dequeue buffer from encoder capture plane" << std::endl;
        ctx->got_error = true;
        return false;
    }
    
    // Push encoded H.264 data to RTSP stream via parent pipeline manager
    if (buffer->planes[0].bytesused > 0) {
        // Get parent pipeline manager from context
        OptimizedPipelineManager* pipelineManager = (OptimizedPipelineManager*)ctx->pipelineManager;
        if (pipelineManager) {
            pipelineManager->pushEncodedData(buffer->planes[0].data, buffer->planes[0].bytesused);
        }
    }
    
    // Queue buffer back to capture plane
    if (ctx->encoder->capture_plane.qBuffer(*v4l2_buf, NULL) < 0) {
        std::cerr << "Error while queueing buffer at capture plane" << std::endl;
        ctx->got_error = true;
        return false;
    }
    
    // Check for EOS
    if (buffer->planes[0].bytesused == 0) {
        return false;
    }
    
    return true;
}

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
    
    OptimizedPipelineManager pipelineManager(sourceAddress, outputAddress, bitrate);
    
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
