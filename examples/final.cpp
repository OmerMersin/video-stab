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

// Global variable for signal handling
volatile sig_atomic_t stopRequested = 0;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down gracefully..." << std::endl;
    stopRequested = 1;
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
    vs::DeepStreamTracker tracker(trackerParams);
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
    FILE* passthroughProcess = nullptr;
    pid_t passthroughPid = -1;
    
    // Frame buffer management for streaming
    const int maxBufferedFrames = 2;  // Keep buffer very small for low latency
    int bufferedFrameCount = 0;

    // Check if we can use passthrough mode (no processing needed)
    bool usePassthrough = !runParams.enhancerEnabled && 
                         !runParams.rollCorrectionEnabled && 
                         !runParams.stabilizationEnabled &&
                         !runParams.trackerEnabled;
    
    if (usePassthrough && videoSource.find("rtsp://") == 0) {
        std::cout << "No processing enabled - using ultra-low latency passthrough mode" << std::endl;
        std::cout << "Make sure MediaMTX is running on port 8554" << std::endl;
        
        // Build FFmpeg passthrough command exactly like your working version
        std::stringstream passthroughCmd;
        passthroughCmd << "ffmpeg -loglevel warning "
                      << "-fflags nobuffer -flags low_delay "
                      << "-rtsp_transport udp "  // Use UDP as in your working command
                      << "-i " << videoSource << " "
                      << "-c copy "  // Copy without re-encoding for minimal latency
                      << "-f rtsp -rtsp_transport tcp "
                      << "rtsp://localhost:8554/forwarded";
        
        std::cout << "Starting passthrough with command: " << passthroughCmd.str() << std::endl;
        
        // Start FFmpeg passthrough process
        passthroughProcess = popen(passthroughCmd.str().c_str(), "r");
        if (!passthroughProcess) {
            std::cerr << "Failed to start passthrough process, falling back to frame processing" << std::endl;
        } else {
            std::cout << "Passthrough mode active - press Ctrl+C to stop" << std::endl;
            
            // Use non-blocking read to check for signals
            fd_set readfds;
            struct timeval timeout;
            int fd = fileno(passthroughProcess);
            char buffer[256];
            
            while (!stopRequested) {
                FD_ZERO(&readfds);
                FD_SET(fd, &readfds);
                timeout.tv_sec = 1;  // 1 second timeout
                timeout.tv_usec = 0;
                
                int result = select(fd + 1, &readfds, NULL, NULL, &timeout);
                
                if (result > 0 && FD_ISSET(fd, &readfds)) {
                    if (fgets(buffer, sizeof(buffer), passthroughProcess) != nullptr) {
                        // Print FFmpeg output for debugging
                        if (strlen(buffer) > 1) {
                            std::cout << "FFmpeg: " << buffer;
                        }
                    } else {
                        // Process ended
                        break;
                    }
                } else if (result < 0) {
                    // Error occurred
                    break;
                }
                
                // Check if we should exit
                if (stopRequested) {
                    std::cout << "Stopping passthrough mode..." << std::endl;
                    break;
                }
            }
            
            int exitCode = pclose(passthroughProcess);
            std::cout << "Passthrough process ended with code: " << exitCode << std::endl;
            return 0;
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
                        
                        if (shouldRestart(usePassthrough, newUsePassthrough)) {
                            std::cout << "Processing mode changed - restart required for optimal performance" << std::endl;
                            std::cout << "New mode: " << (newUsePassthrough ? "Passthrough" : "Processing") << std::endl;
                            // Note: We continue with current mode for now, full restart would be needed for mode switch
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
        if (runParams.trackerEnabled) {
            // Process frame through tracker
            auto detections = tracker.processFrame(*framePtr);
            
            // Check for new tracking coordinates from TCP
            if (tcp.tryGetLatest(x, y)) {
                std::cout << "Received tracking coordinates: (" << x << "," << y << ") with " 
                          << detections.size() << " detections available" << std::endl;
                
                // Draw detections with the selected coordinates
                cv::Mat trackedFrame = tracker.drawDetections(*framePtr, detections, x, y);
                if (framePtr == &tempFrame3) {
                    tempFrame3 = trackedFrame;  // Update the existing frame
                } else {
                    tempFrame3 = trackedFrame;  // Use tempFrame3 for tracked output
                    framePtr = &tempFrame3;
                }
            } else {
                // No new coordinates, use previous selection
                cv::Mat trackedFrame = tracker.drawDetections(*framePtr, detections, -1, -1);
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
            // Skip frame sending if we're behind to catch up
            static auto lastSendTime = std::chrono::high_resolution_clock::now();
            auto currentTime = std::chrono::high_resolution_clock::now();
            double timeSinceLastSend = std::chrono::duration<double, std::milli>(currentTime - lastSendTime).count();
            
            // Only send frame if enough time has passed (adaptive frame dropping)
            double targetInterval = 1000.0 / fps;
            if (timeSinceLastSend >= targetInterval * 0.8) {  // Allow 20% tolerance
                // Ensure frame is in BGR format and correct size
                cv::Mat outputFrame;
                if (processedFrame.cols != frameWidth || processedFrame.rows != frameHeight) {
                    cv::resize(processedFrame, outputFrame, cv::Size(frameWidth, frameHeight), 0, 0, cv::INTER_LINEAR);
                } else if (!processedFrame.isContinuous()) {
                    processedFrame.copyTo(outputFrame);
                } else {
                    outputFrame = processedFrame;
                }
                
                // Send to GStreamer pipeline  
                GstBuffer* buf = gst_buffer_new_allocate(
                    nullptr, outputFrame.total() * outputFrame.elemSize(), nullptr);
                GstMapInfo map;
                gst_buffer_map(buf, &map, GST_MAP_WRITE);
                std::memcpy(map.data, outputFrame.data, map.size);
                gst_buffer_unmap(buf, &map);
                GST_BUFFER_PTS(buf) = gst_util_uint64_scale(frameCounter++, GST_SECOND, fps);
                gst_app_src_push_buffer(appsrc, buf);
                
                lastSendTime = currentTime;
                bufferedFrameCount = 0;  // Reset buffer count on successful send
            } else {
                bufferedFrameCount++;
                // Drop frames if we're getting too far behind
                if (bufferedFrameCount > maxBufferedFrames) {
                    bufferedFrameCount = 0;  // Reset and skip this frame
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
    
    // Cleanup passthrough process
    if (passthroughProcess) {
        std::cout << "Closing passthrough process..." << std::endl;
        int exitCode = pclose(passthroughProcess);
        std::cout << "Passthrough process closed with code: " << exitCode << std::endl;
        passthroughProcess = nullptr;
    }
    
    cv::destroyAllWindows();
    std::cout << "Cleanup complete." << std::endl;
    return 0;
}
