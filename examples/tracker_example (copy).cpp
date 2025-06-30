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
    
    // Build the FFmpeg command with optimized parameters
    std::stringstream ffmpegCmd;
    ffmpegCmd << "ffmpeg -loglevel warning "
              << "-re -f rawvideo -pixel_format bgr24 "
              << "-video_size " << frameWidth << "x" << frameHeight << " "
              << "-framerate " << fps << " "
              << "-i - "
              << "-c:v libx264 "
              << "-preset veryfast " // Better balance of quality and speed than ultrafast
              << "-tune zerolatency "
              << "-profile:v baseline "
              << "-x264-params keyint=" << int(fps*2) << ":min-keyint=" << int(fps) << " "
              << "-b:v " << bitrate << "k "
              << "-maxrate " << bitrate*1.5 << "k "
              << "-bufsize " << bitrate << "k "
              << "-pix_fmt yuv420p " // Ensure compatibility
              << "-f rtsp "
              << "-rtsp_transport tcp " // More reliable than UDP for local streaming
              << "rtsp://localhost:8554/forwarded";
    
    std::cout << "Starting FFmpeg with command: " << ffmpegCmd.str() << std::endl;
    
    FILE* ffmpeg = popen(ffmpegCmd.str().c_str(), "w");
    
    if (!ffmpeg) {
        std::cerr << "Failed to start FFmpeg process" << std::endl;
        return 1;
    }


    int delayMs = (runParams.optimizeFps) ? 1 : static_cast<int>(1000.0 / fps);

    const int windowWidth = runParams.width;
    const int windowHeight = runParams.height;
if (!headless) {
    cv::namedWindow("Raw", cv::WINDOW_NORMAL);
    cv::resizeWindow("Raw", windowWidth, windowHeight);

    cv::namedWindow("Final", cv::WINDOW_NORMAL);
    cv::resizeWindow("Final", windowWidth, windowHeight);
}
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
                    
                    // 4. Restart FFmpeg if encoding parameters changed
                    int newBitrate = (frameWidth * frameHeight * fps * 0.1) / 1000;
                    if (newBitrate < 800) newBitrate = 800;
                    if (newBitrate > 8000) newBitrate = 8000;
                    
                    if (newBitrate != bitrate) {
                        bitrate = newBitrate;
                        
                        // Rebuild FFmpeg command
                        std::stringstream newFfmpegCmd;
                        newFfmpegCmd << "ffmpeg -loglevel warning "
                                  << "-re -f rawvideo -pixel_format bgr24 "
                                  << "-video_size " << frameWidth << "x" << frameHeight << " "
                                  << "-framerate " << fps << " "
                                  << "-i - "
                                  << "-c:v libx264 "
                                  << "-preset veryfast "
                                  << "-tune zerolatency "
                                  << "-profile:v baseline "
                                  << "-x264-params keyint=" << int(fps*2) << ":min-keyint=" << int(fps) << " "
                                  << "-b:v " << bitrate << "k "
                                  << "-maxrate " << bitrate*1.5 << "k "
                                  << "-bufsize " << bitrate << "k "
                                  << "-pix_fmt yuv420p "
                                  << "-f rtsp "
                                  << "-rtsp_transport tcp "
                                  << "rtsp://localhost:8554/forwarded";
                        
                        std::cout << "Restarting FFmpeg with updated parameters" << std::endl;
                        
                        // Close and reopen FFmpeg
                        if (ffmpeg) {
                            pclose(ffmpeg);
                        }
                        ffmpeg = popen(newFfmpegCmd.str().c_str(), "w");
                        if (!ffmpeg) {
                            std::cerr << "Failed to restart FFmpeg process" << std::endl;
                        }
                        
                        // Update the command string for potential future restarts
                        ffmpegCmd.str("");
                        ffmpegCmd << newFfmpegCmd.str();
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
	    frame = tracker.drawDetections(frame, detections);
	    
	    // Optionally, print detection count for debugging
	    if (trackerParams.debugMode) {
		std::cout << "Detected " << detections.size() << " objects" << std::endl;
	    }
	}


        // rtspServer.pushFrame(frame);
                // Display final output
        if (!frame.empty()) {
            if (!headless){
                cv::imshow("Final", frame);
            }
            // Ensure frame has the correct dimensions for FFmpeg
            if (frame.cols != frameWidth || frame.rows != frameHeight && !headless) {
                cv::resize(frame, frame, cv::Size(frameWidth, frameHeight));
            }
            
            // Make sure frame is in BGR24 format for FFmpeg
            if (frame.type() != CV_8UC3) {
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
            }
            
            if (ffmpeg) {
                // Check if pipe is still valid
                static int errorCount = 0;
                size_t writeSize = frame.total() * frame.elemSize();
                size_t written = fwrite(frame.data, 1, writeSize, ffmpeg);
                
                if (written != writeSize) {
                    errorCount++;
                    std::cerr << "FFmpeg pipe error: written " << written << " of " << writeSize 
                              << " bytes. Error count: " << errorCount << std::endl;
                    if (errorCount > 5) {
                        std::cerr << "FFmpeg pipe error, restarting..." << std::endl;
                        pclose(ffmpeg);
                        ffmpeg = popen(ffmpegCmd.str().c_str(), "w");
                        errorCount = 0;
                    }
                } else {
                    // Ensure data is sent immediately
                    fflush(ffmpeg);
                    errorCount = 0;  // Reset error count on successful write
                }
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
    if (ffmpeg) {
        pclose(ffmpeg);
    }
    if (!headless) {
    	cv::destroyAllWindows();
    }
    return 0;
}
