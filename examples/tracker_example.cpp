#include "video/RollCorrection.h"
#include "video/CamCap.h"
#include "video/AutoZoomCrop.h"
#include "video/Stabilizer.h"
#include "video/Mode.h"
#include "video/Enhancer.h"
#include "video/RTSPServer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <X11/Xlib.h>
#include <cstdio>   // for popen, pclose
#include <sys/stat.h>
#include "video/DeepStreamTracker.h"
#include "video/TcpReciever.h"
#include <gst/app/gstappsrc.h>

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

// Read Stabilizer Parameters (with SightLine-inspired enhancements)
    cv::FileNode stabNode = fs["stabilizer"];
    if (!stabNode.empty()) {
        // Basic parameters (original)
        stabNode["smoothing_radius"] >> stabParams.smoothingRadius;
        stabNode["border_type"] >> stabParams.borderType;
        stabNode["border_size"] >> stabParams.borderSize;
        stabNode["crop_n_zoom"] >> stabParams.cropNZoom;
        stabNode["logging"] >> stabParams.logging;
        stabNode["use_cuda"] >> stabParams.useCuda;
        stabNode["max_corners"] >> stabParams.maxCorners;
        stabNode["quality_level"] >> stabParams.qualityLevel;
        stabNode["min_distance"] >> stabParams.minDistance;
        stabNode["block_size"] >> stabParams.blockSize;
        
        // SightLine-inspired parameters
        std::string smoothingMethod;
        stabNode["smoothing_method"] >> smoothingMethod;
        if (!smoothingMethod.empty()) {
            stabParams.smoothingMethod = smoothingMethod;
        }
        
        // Adaptive smoothing
        stabNode["adaptive_smoothing"] >> stabParams.adaptiveSmoothing;
        stabNode["min_smoothing_radius"] >> stabParams.minSmoothingRadius;
        stabNode["max_smoothing_radius"] >> stabParams.maxSmoothingRadius;
        
        // Outlier rejection
        stabNode["outlier_rejection"] >> stabParams.outlierRejection;
        stabNode["outlier_threshold"] >> stabParams.outlierThreshold;
        
        // Gaussian smoothing
        stabNode["gaussian_sigma"] >> stabParams.gaussianSigma;
        
        // Motion prediction
        stabNode["motion_prediction"] >> stabParams.motionPrediction;
        stabNode["intentional_motion_threshold"] >> stabParams.intentionalMotionThreshold;
        
        // ROI
        stabNode["use_roi"] >> stabParams.useROI;
        if (stabParams.useROI) {
            int x = 0, y = 0, width = 0, height = 0;
            stabNode["roi_x"] >> x;
            stabNode["roi_y"] >> y;
            stabNode["roi_width"] >> width;
            stabNode["roi_height"] >> height;
            if (width > 0 && height > 0) {
                stabParams.roi = cv::Rect(x, y, width, height);
            }
        }
        
        // Horizon lock
        stabNode["horizon_lock"] >> stabParams.horizonLock;
        
        // Feature detector
        int detectorType = 0;
        stabNode["feature_detector_type"] >> detectorType;
        stabParams.featureDetector = static_cast<vs::Stabilizer::Parameters::FeatureDetector>(detectorType);
        
        // Feature detector parameters
        stabNode["fast_threshold"] >> stabParams.fastThreshold;
        stabNode["orb_features"] >> stabParams.orbFeatures;
        
        stabNode["fade_duration"] >> stabParams.fadeDuration;
	stabNode["fade_alpha"] >> stabParams.fadeAlpha;
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
// Update your readConfig function to use the new Parameters structure
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
    if (!XInitThreads()) {
        std::cerr << "XInitThreads() failed." << std::endl;
        return 1;
    }

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>\n";
        return 1;
    }
    std::string configFile = argv[1];
    
    bool headless = (std::getenv("DISPLAY") == nullptr);

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

    vs::CamCap cam(camParams);
    cam.start();
    
    vs::DeepStreamTracker tracker(trackerParams);


    // 2) Start your RTSP server
    // RTSPServer rtspServer;
    // // For example, run on TCP port 8554 at path "/test"
    // if(!rtspServer.startServer(8554, "/test")) {
    //     std::cerr << "Failed to start RTSP server." << std::endl;
    //     return 1;
    // }
    // Make sure to get the ACTUAL frame dimensions before setting up FFmpeg
    double fps = cam.getFrameRate();
    if (fps < 1.0) fps = 30.0;
    std::cout << "Video framerate: " << fps << " FPS" << std::endl;
    
    // Read a test frame to get actual dimensions
    cv::Mat testFrame = cam.read();
    int frameWidth = testFrame.cols;
    int frameHeight = testFrame.rows;
    
    // Override with config values if specified
    if (runParams.width > 0 && runParams.height > 0) {
        frameWidth = runParams.width;
        frameHeight = runParams.height;
    }
    
    std::cout << "Frame dimensions: " << frameWidth << "x" << frameHeight << std::endl;
    
        // Calculate appropriate bitrate based on resolution
    int bitrate = (frameWidth * frameHeight * fps * 0.1) / 1000; // In Kbps
    if (bitrate < 800) bitrate = 800; // Minimum bitrate floor
    if (bitrate > 8000) bitrate = 8000; // Maximum bitrate ceiling
GstElement* pipeline = nullptr;
GstAppSrc*  appsrc   = nullptr;

auto buildStreamer = [&]() {
    gst_init(nullptr,nullptr);

    std::string pipe =
        "appsrc name=src is-live=true format=time block=false "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(frameWidth) +
        ",height="  + std::to_string(frameHeight) +
        ",framerate=" + std::to_string(int(fps)) + "/1 ! "
        "queue max-size-buffers=10 leaky=downstream ! "
        "videoconvert ! video/x-raw,format=NV12 ! "
        "x264enc threads=4 tune=zerolatency speed-preset=ultrafast "
        "bitrate=" + std::to_string(bitrate) + " ! "
        "rtspclientsink location=rtsp://localhost:8554/forwarded protocols=tcp";

    pipeline = gst_parse_launch(pipe.c_str(), nullptr);
    appsrc   = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(pipeline), "src"));
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
};

buildStreamer();


    int delayMs = (runParams.optimizeFps) ? 1 : static_cast<int>(1000.0 / fps);

    const int windowWidth = runParams.width;
    const int windowHeight = runParams.height;
if (!headless) {
    cv::namedWindow("Raw", cv::WINDOW_NORMAL);
    cv::resizeWindow("Raw", windowWidth, windowHeight);

    cv::namedWindow("Final", cv::WINDOW_NORMAL);
    cv::resizeWindow("Final", windowWidth, windowHeight);
}
uint64_t frameCounter = 0;

vs::TcpReciever tcp(5000);   // listen on port 5000
tcp.start();
int x = -1, y = -1;

    while (true) {
        // Check if the config file has been modified
        if (stat(configFile.c_str(), &configStat) == 0) {
            if (configStat.st_mtime != lastConfigModTime) {
                std::cout << "Configuration file updated, reloading parameters..." << std::endl;
                if (readConfig(configFile, videoSource, runParams, enhancerParams, rollParams, stabParams, camParams, trackerParams)) {
                    std::cout << "Configuration reloaded successfully." << std::endl;
                    lastConfigModTime = configStat.st_mtime;
                    
                    // Update components with new parameters
                    
                    // 1. Update Stabilizer with new parameters
                     try {
                        // Create a new stabilizer with the updated parameters
                        vs::Stabilizer newStabilizer(stabParams);
                        
                        // Now clean the old stabilizer before it goes out of scope
                        stab.clean();
                        
                        // Swap the old one with the new one (avoids explicit destruction)
                        std::swap(stab, newStabilizer);
                        
                        // newStabilizer will now be destructed safely when it goes out of scope
                    } catch (const std::exception& e) {
                        std::cerr << "Error recreating stabilizer: " << e.what() << std::endl;
                    }
                    
                    // 2. Update camera if source changed
                    if (camParams.source != videoSource) {
                        std::cout << "Video source changed from " << videoSource << " to " << camParams.source << std::endl;
                        videoSource = camParams.source;
                        
                        // Stop the current camera
                        cam.stop();
                        
                        // Instead of assignment, destroy and reconstruct
                        cam.~CamCap();  // Explicitly call destructor
                        new (&cam) vs::CamCap(camParams);  // Placement new to reconstruct
                        
                        cam.start();
                        
                        // Get fresh frame rate after camera restart
                        fps = cam.getFrameRate();
                        if (fps < 1.0) fps = 30.0;
                        std::cout << "Updated video framerate: " << fps << " FPS" << std::endl;
                        
                        // Update delay time for main loop
                        delayMs = (runParams.optimizeFps) ? 1 : static_cast<int>(1000.0 / fps);
                    }
                    
                    // 3. Update window sizes if dimensions changed
                    if (runParams.width > 0 && runParams.height > 0 && 
                        (runParams.width != windowWidth || runParams.height != windowHeight)) {
                        
                        cv::resizeWindow("Raw", runParams.width, runParams.height);
                        cv::resizeWindow("Final", runParams.width, runParams.height);
                        
                        // Update dimension tracking
                        frameWidth = runParams.width;
                        frameHeight = runParams.height;
                    }
                    
                    
                    std::cout << "All components updated with new configuration" << std::endl;
                } else {
                    std::cerr << "Failed to reload configuration." << std::endl;
                }
            }
        }

        auto startTime = std::chrono::high_resolution_clock::now();  // Start timer
        cv::Mat frame = cam.read();
        if (!frame.empty() && !headless) {
            cv::imshow("Raw", frame);
        }

        if (runParams.enhancerEnabled) {
            frame = vs::Enhancer::enhanceImage(frame, enhancerParams);
        }

        // Apply Roll Correction
        if (runParams.rollCorrectionEnabled) {
            frame = vs::RollCorrection::autoCorrectRoll(frame, rollParams);
        }

        // Apply Stabilization
        if (runParams.stabilizationEnabled) {
            frame = stab.stabilize(frame);
        }
        
if (runParams.trackerEnabled) {
    // Process frame through tracker
    auto detections = tracker.processFrame(frame);
    
    // Draw detections on the frame
    if (tcp.tryGetLatest(x, y)) {
        std::cout << "Received coordinates: (" << x << "," << y << ") with " 
                  << detections.size() << " detections available" << std::endl;
        
        // Pass the coordinates directly to drawDetections
        frame = tracker.drawDetections(frame, detections, x, y);
    } else {
        // No new coordinates, use previous selection
        frame = tracker.drawDetections(frame, detections, -1, -1);
    }
}


        // rtspServer.pushFrame(frame);
                // Display final output
        if (!frame.empty()) {
            if (!headless){
                cv::imshow("Final", frame);
            }
            // Ensure frame has the correct dimensions for FFmpeg
            if (frame.cols != frameWidth || frame.rows != frameHeight) {
                cv::resize(frame, frame, cv::Size(frameWidth, frameHeight));
            }
            
            // Make sure frame is in BGR24 format for FFmpeg
            if (frame.type() != CV_8UC3) {
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
            }
            
            if (appsrc)
{
    GstBuffer* buf = gst_buffer_new_allocate(
        nullptr, frame.total() * frame.elemSize(), nullptr);
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_WRITE);
    std::memcpy(map.data, frame.data, map.size);
    gst_buffer_unmap(buf,&map);
    GST_BUFFER_PTS(buf) = gst_util_uint64_scale(frameCounter++, GST_SECOND, fps);
    gst_app_src_push_buffer(appsrc, buf);
}
        }
        

    // --- Measure Processing Time ---
        auto endTime = std::chrono::high_resolution_clock::now();
        double frameTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // Time in ms
        double fps = 1000.0 / frameTime;  // Convert to FPS

        std::cout << "Processing Time: " << frameTime << " ms | FPS: " << fps << std::endl;

                
if (!headless) {
    if (cv::waitKey(delayMs) == 27) break;
} else {
    // no GUI: just sleep for the same delay
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
}
    }
    if (appsrc) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
	gst_object_unref(pipeline);

    }
    if (!headless) {
    	cv::destroyAllWindows();
    }
    return 0;
}
