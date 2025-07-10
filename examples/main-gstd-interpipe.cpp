#include "video/RollCorrection.h"
#include "video/CamCapInterpipe.h"
#include "video/GstdManager.h"
#include "video/AutoZoomCrop.h"
#include "video/Stabilizer.h"
#include "video/Mode.h"
#include "video/Enhancer.h"
#include "video/DeepStreamTracker.h"
#include "video/TcpReciever.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>
#include <memory>

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

int main(int argc, char** argv) {
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config_file> <rtsp_output_url>\n";
        std::cerr << "Example: " << argv[0] << " config.yaml rtsp://localhost:8554/stream\n";
        return 1;
    }

    std::string configFile = argv[1];
    std::string rtspOutput = argv[2];

    // Store the last modification time for config file monitoring
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
    vs::DeepStreamTracker::Parameters trackerParams;
    
    // Set default tracker parameters
    trackerParams.modelEngine = "/opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/resnet18_trafficcamnet.etlt_b8_gpu0_int8.engine";
    trackerParams.modelConfigFile = "/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_infer_primary_resnet18.txt";
    trackerParams.processingWidth = 640;
    trackerParams.processingHeight = 368;
    trackerParams.confidenceThreshold = 0.3;

    // Read the config file
    if (!readConfig(configFile, videoSource, runParams, enhancerParams, rollParams, stabParams, trackerParams)) {
        return 1;
    }

    std::cout << "Using video source: " << videoSource << std::endl;
    std::cout << "Output RTSP URL: " << rtspOutput << std::endl;
    std::cout << "Roll Correction: " << (runParams.rollCorrectionEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Stabilizer: " << (runParams.stabilizationEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Tracker: " << (runParams.trackerEnabled ? "Enabled" : "Disabled") << std::endl;

    // Initialize GStreamer daemon manager
    vs::GstdManager::Parameters gstdParams;
    gstdParams.rtspSource = videoSource;
    gstdParams.rtspOutput = rtspOutput;
    gstdParams.width = runParams.width > 0 ? runParams.width : 1920;
    gstdParams.height = runParams.height > 0 ? runParams.height : 1080;
    gstdParams.fps = 30;
    gstdParams.bitrate = 4000;
    gstdParams.logging = true;

    auto gstdManager = std::make_unique<vs::GstdManager>(gstdParams);

    // Initialize and start gstd manager
    if (!gstdManager->initialize()) {
        std::cerr << "Failed to initialize GStreamer daemon manager" << std::endl;
        return 1;
    }

    if (!gstdManager->start()) {
        std::cerr << "Failed to start GStreamer daemon manager" << std::endl;
        return 1;
    }

    // Initialize interpipe camera for processing mode
    vs::CamCapInterpipe::Parameters camParams;
    camParams.interpipeInputName = "processing_out";
    camParams.interpipeOutputName = "processed_out";
    camParams.width = gstdParams.width;
    camParams.height = gstdParams.height;
    camParams.fps = gstdParams.fps;
    camParams.logging = true;

    auto cam = std::make_unique<vs::CamCapInterpipe>(camParams);

    // Initialize processing modules
    vs::Stabilizer stab(stabParams);
    
    // Initialize tracker and TCP receiver
    std::unique_ptr<vs::DeepStreamTracker> tracker;
    if (runParams.trackerEnabled) {
        tracker = std::make_unique<vs::DeepStreamTracker>(trackerParams);
    }
    
    vs::TcpReciever tcp(5000);
    tcp.start();
    int x = -1, y = -1;

    // Determine initial mode
    bool usePassthrough = !runParams.enhancerEnabled && 
                         !runParams.rollCorrectionEnabled && 
                         !runParams.stabilizationEnabled &&
                         !runParams.trackerEnabled;

    std::cout << "Starting in " << (usePassthrough ? "passthrough" : "processing") << " mode" << std::endl;

    // Set initial mode
    if (usePassthrough) {
        gstdManager->switchToPassthrough();
    } else {
        gstdManager->switchToProcessing();
        
        // Initialize and start interpipe camera for processing
        if (!cam->initialize()) {
            std::cerr << "Failed to initialize interpipe camera" << std::endl;
            return 1;
        }
        cam->start();
    }

    // Create windows for display if not optimizing for FPS
    int windowWidth = runParams.width > 0 ? runParams.width : 640;
    int windowHeight = runParams.height > 0 ? runParams.height : 480;
    
    if (!runParams.optimizeFps) {
        cv::namedWindow("Processing", cv::WINDOW_NORMAL);
        cv::resizeWindow("Processing", windowWidth, windowHeight);
    }

    std::cout << "Ultra-low latency gstd-based streaming system ready!" << std::endl;
    std::cout << "Passthrough mode: Zero additional latency" << std::endl;
    std::cout << "Processing mode: Minimal latency with full processing pipeline" << std::endl;

    // Main processing loop
    int frameCounter = 0;
    int configCheckCounter = 0;
    
    while (!stopRequested) {
        // Check for config file changes periodically
        if (configCheckCounter++ % 150 == 0) {  // Check every ~5 seconds at 30fps
            if (stat(configFile.c_str(), &configStat) == 0) {
                if (configStat.st_mtime != lastConfigModTime) {
                    std::cout << "Config file changed, reloading..." << std::endl;
                    lastConfigModTime = configStat.st_mtime;
                    
                    // Reload config
                    vs::Mode::Parameters newRunParams;
                    vs::Enhancer::Parameters newEnhancerParams;
                    vs::RollCorrection::Parameters newRollParams;
                    vs::Stabilizer::Parameters newStabParams;
                    vs::DeepStreamTracker::Parameters newTrackerParams;
                    std::string newVideoSource;
                    
                    if (readConfig(configFile, newVideoSource, newRunParams, newEnhancerParams, 
                                 newRollParams, newStabParams, newTrackerParams)) {
                        
                        // Check if we need to switch modes
                        bool newUsePassthrough = !newRunParams.enhancerEnabled && 
                                               !newRunParams.rollCorrectionEnabled && 
                                               !newRunParams.stabilizationEnabled &&
                                               !newRunParams.trackerEnabled;
                        
                        if (newUsePassthrough != usePassthrough) {
                            std::cout << "Switching to " << (newUsePassthrough ? "passthrough" : "processing") << " mode" << std::endl;
                            
                            if (newUsePassthrough) {
                                // Switch to passthrough mode
                                if (cam) {
                                    cam->stop();
                                }
                                gstdManager->switchToPassthrough();
                            } else {
                                // Switch to processing mode
                                if (!cam) {
                                    cam = std::make_unique<vs::CamCapInterpipe>(camParams);
                                    cam->initialize();
                                }
                                if (!cam->isHealthy()) {
                                    cam->start();
                                }
                                gstdManager->switchToProcessing();
                            }
                            
                            usePassthrough = newUsePassthrough;
                        }
                        
                        // Update parameters
                        runParams = newRunParams;
                        enhancerParams = newEnhancerParams;
                        rollParams = newRollParams;
                        stabParams = newStabParams;
                        trackerParams = newTrackerParams;
                        
                        std::cout << "Configuration updated successfully" << std::endl;
                    }
                }
            }
        }

        // Process frames only in processing mode
        if (!usePassthrough && cam && cam->isHealthy()) {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            cv::Mat frame = cam->read();
            if (frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Apply processing pipeline
            cv::Mat* framePtr = &frame;
            cv::Mat tempFrame1, tempFrame2, tempFrame3, tempFrame4;

            // Apply Enhancement
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
                if (!tempFrame3.empty()) {
                    framePtr = &tempFrame3;
                }
            }

            // Apply Tracking
            if (runParams.trackerEnabled && tracker) {
                // Get tracking coordinates from TCP
                int tcpX, tcpY;
                if (tcp.tryGetLatest(tcpX, tcpY)) {
                    x = tcpX;
                    y = tcpY;
                }
                
                // Apply tracking - get detections and draw them
                auto detections = tracker->processFrame(*framePtr);
                tempFrame4 = tracker->drawDetections(*framePtr, detections, x, y);
                framePtr = &tempFrame4;
            }

            // Send processed frame back to interpipe
            cam->write(*framePtr);

            // Display frame occasionally
            if (!runParams.optimizeFps && frameCounter % 30 == 0) {
                cv::Mat displayFrame;
                if (framePtr->cols != windowWidth || framePtr->rows != windowHeight) {
                    cv::resize(*framePtr, displayFrame, cv::Size(windowWidth, windowHeight));
                } else {
                    displayFrame = *framePtr;
                }
                cv::imshow("Processing", displayFrame);
            }

            // Performance monitoring
            auto endTime = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            
            if (frameCounter % 300 == 0) {
                std::cout << "Processing time: " << frameTime << "ms" << std::endl;
            }
        } else if (usePassthrough) {
            // In passthrough mode, just keep the application alive
            if (frameCounter % 3000 == 0) {  // Every ~100 seconds at 30fps
                std::cout << "Passthrough mode: System running..." << std::endl;
            }
        }

        frameCounter++;

        // Key detection for manual control
        if (!runParams.optimizeFps && frameCounter % 10 == 0) {
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {  // 'q' or ESC
                break;
            } else if (key == 'p') {
                // Toggle passthrough mode
                if (usePassthrough) {
                    std::cout << "Switching to processing mode..." << std::endl;
                    if (!cam) {
                        cam = std::make_unique<vs::CamCapInterpipe>(camParams);
                        cam->initialize();
                    }
                    if (!cam->isHealthy()) {
                        cam->start();
                    }
                    gstdManager->switchToProcessing();
                    usePassthrough = false;
                } else {
                    std::cout << "Switching to passthrough mode..." << std::endl;
                    if (cam) {
                        cam->stop();
                    }
                    gstdManager->switchToPassthrough();
                    usePassthrough = true;
                }
            }
        }

        // Small delay to prevent excessive CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30fps
    }

    std::cout << "Shutting down..." << std::endl;

    // Cleanup
    if (cam) {
        cam->stop();
    }
    
    tcp.stop();
    gstdManager->stop();
    gstdManager->cleanup();

    cv::destroyAllWindows();
    
    std::cout << "Cleanup complete." << std::endl;
    return 0;
}
