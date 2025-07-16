// Robust Digital Stabilizer - Focused on Shake Avoidance
// Optimized for Jetson Orin Nano
// --------------------------------------------------------

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <deque>
#include <vector>
#include <numeric>
#include "video/Stabilizer.h"

// GPU includes if available
#if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#ifdef HAVE_OPENCV_CUDAOPTFLOW
#include <opencv2/cudaoptflow.hpp>
#endif

namespace vs {
    /**
     * @brief GPU/CPU Video Stabilizer replicates vidgear's stabilizer logic.
     */

    // Helper: map borderType -> OpenCV border mode
    static int mapBorderMode(const std::string &borderType) {
        if (borderType == "reflect")      return cv::BORDER_REFLECT;
        if (borderType == "reflect_101")  return cv::BORDER_REFLECT_101;
        if (borderType == "replicate")    return cv::BORDER_REPLICATE;
        if (borderType == "wrap")         return cv::BORDER_WRAP;
        if (borderType == "fade")         return -1; // Special value for fade effect
        return cv::BORDER_CONSTANT; // default black
    }
    
    void Stabilizer::logMessage(const std::string& msg, bool isError) const {
            if (isError) {
                std::cerr << "[ERROR] " << msg << std::endl;
            } else {
                std::cout << "[INFO] " << msg << std::endl;
            }
        }


    // Constructor
    Stabilizer::Stabilizer(const Parameters &params)
    : params_(params)
    {
        if(params_.logging) {
            logMessage("=== STABILIZER PARAMETER DEBUG ===", false);
            logMessage("droneHighFreqMode: " + std::string(params_.droneHighFreqMode ? "TRUE" : "FALSE"), false);
            logMessage("enableVirtualCanvas: " + std::string(params_.enableVirtualCanvas ? "TRUE" : "FALSE"), false);
            logMessage("hfDeadZoneThreshold: " + std::to_string(params_.hfDeadZoneThreshold), false);
            logMessage("hfFreezeDuration: " + std::to_string(params_.hfFreezeDuration), false);
            logMessage("hfMotionAccumulatorDecay: " + std::to_string(params_.hfMotionAccumulatorDecay), false);
            logMessage("=== END PARAMETER DEBUG ===", false);
            logMessage("Initializing Jetson Orin Nano optimized stabilizer...", false);
        }

        useGpu_ = params_.useCuda;
        borderMode_ = mapBorderMode(params_.borderType);

        if(params_.cropNZoom && params_.borderType != "black") {
            // force black if crop+zoom
            if(params_.logging) logMessage("cropNZoom => ignoring borderType, using black.", false);
            borderMode_ = cv::BORDER_CONSTANT;
        }
        
        // Initialize fade effect variables if using fade border
        if (params_.borderType == "fade") {
            fadeFrameCount_ = 0;
            borderHistory_ = cv::Mat();
            if(params_.logging) logMessage("Using fade border effect", false);
        }

        // Professional-grade stabilization parameters
        int effectiveRadius = std::max(5, std::min(params_.smoothingRadius, 25));
        
        // Multi-stage smoothing kernels
        boxKernel_.resize(effectiveRadius, 1.0f / effectiveRadius);
        
        // Enhanced motion analysis buffers for professional-grade stabilization
        transforms_.reserve(500);
        path_.reserve(500);
        smoothedPath_.reserve(500);
        motionHistory_.reserve(100);  // For motion pattern analysis
        intentionalMotionBuffer_.reserve(50);  // For intent detection
        
        // Initialize professional stabilization components
        initializeRobustStabilization();

        // Jetson Orin Nano optimized initialization
        // Skip CLAHE for real-time performance - use simple histogram equalization instead
        
    #if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
        if(useGpu_) {
            // Create multiple CUDA streams for parallel processing
            cudaStream_ = cv::cuda::Stream();  // Primary stream
            analysisStream_ = cv::cuda::Stream();  // Analysis stream
            warpStream_ = cv::cuda::Stream();  // Warping stream
            
            // Pre-allocate GPU memory buffers - increased sizes for 1920x1080 frames
            gpuFrameBuffer_.create(2160, 3840, CV_8UC3);  // 4K max for safety
            gpuGrayBuffer_.create(540, 960, CV_8UC1);     // Analysis resolution buffer
            gpuPrevGrayBuffer_.create(540, 960, CV_8UC1);
            gpuWarpBuffer_.create(2160, 3840, CV_8UC3);   // Match frame buffer
            
            if(params_.logging) logMessage("Jetson Orin Nano GPU optimization enabled with multi-stream processing");
            
            // Apply Jetson-specific optimizations
            optimizeForJetson();
        }
    #endif

    #ifdef HAVE_OPENCV_CUDAOPTFLOW
        if(useGpu_) {
            // Jetson optimized optical flow parameters
            pyrLK_ = cv::cuda::SparsePyrLKOpticalFlow::create();
            pyrLK_->setWinSize(cv::Size(15,15));  // Reduced window size for speed
            pyrLK_->setMaxLevel(2);               // Reduced pyramid levels
            pyrLK_->setNumIters(20);              // Reduced iterations
            
            // Use Jetson-optimized GFTT parameters
    #ifdef HAVE_OPENCV_CUDAFEATURES2D
            gfttDetector_ = cv::cuda::createGoodFeaturesToTrackDetector(
                CV_8UC1, 
                std::min(params_.maxCorners, 300),  // Limit features for real-time
                0.02,     // Slightly lower quality for speed
                10.0,     // Increased min distance
                3,        // Smaller block size
                false,    // useHarris = false (faster)
                0.04
            );
    #endif
        }
    #endif
    
        // HF: Initialize drone high-frequency mode components
        if (params_.droneHighFreqMode) {
            hfTranslationHistory_.clear();
            hfMedianTranslation_ = cv::Vec2f(0.0f, 0.0f);
            hfRotationLowPass_ = 0.0f;
            hfFeatureStarvationCounter_ = 0;
            
            // HF: Initialize dead zone freeze shot state
            hfInDeadZone_ = false;
            hfFreezeCounter_ = 0;
            hfMotionAccumulator_ = 0.0f;
            hfFrozenTransform_ = Transform(0.0f, 0.0f, 0.0f);
            
            // Initialize conditional CLAHE for feature enhancement
            if (params_.enableConditionalCLAHE) {
                hfConditionalCLAHE_ = cv::createCLAHE(2.0, cv::Size(8, 8));
            }
            
            if (params_.logging) {
                logMessage("Drone high-frequency mode enabled - prop vibration suppression active", false);
            }
        }
    }

    // Professional-grade stabilization system initialization (iPhone/GoPro-like)
    void Stabilizer::initializeRobustStabilization() {
        // Initialize motion prediction system
        velocityFilter_.resize(5, 0.0f);  // Velocity smoothing
        accelerationFilter_.resize(3, 0.0f);  // Acceleration smoothing
        
        // Horizon lock initialization
        horizonLockEnabled_ = true;
        horizonAngle_ = 0.0f;
        horizonConfidence_ = 0.0f;
        
        // Motion analysis thresholds (iPhone/GoPro-like)
        intentionalMotionThreshold_ = 15.0f;  // degrees/second
        shakeLevelThreshold_ = 2.0f;  // pixels
        
        // Adaptive smoothing parameters
        minSmoothingStrength_ = 0.3f;
        maxSmoothingStrength_ = 0.95f;
        
        // Scene classification
        lastSceneType_ = SceneType::NORMAL;
        sceneChangeCounter_ = 0;
        
        // Initialize multi-frame motion model
        frameCount_ = 0;
        lastValidTransform_ = Transform();
        
        // Professional motion history tracking
        motionQuality_ = 0.8f;
        stabilizationStrength_ = 0.7f;
        
        // Initialize Virtual Canvas Stabilization if enabled
        if (params_.enableVirtualCanvas) {
            temporalFrameBuffer_.clear();
            temporalTransformBuffer_.clear();
            // Note: std::deque doesn't have reserve(), but that's okay - it will grow as needed
            
            currentCanvasScale_ = params_.canvasScaleFactor;
            virtualCanvasFrameCount_ = 0;
            
            if(params_.logging) {
                logMessage("Virtual Canvas Stabilization enabled - Canvas scale: " + 
                          std::to_string(currentCanvasScale_) + 
                          ", Buffer size: " + std::to_string(params_.temporalBufferSize), false);
            }
        }
        
        if(params_.logging) logMessage("Professional stabilization system initialized", false);
    }

    Stabilizer::~Stabilizer()
    {
        clean();
    }

    void Stabilizer::clean()
    {
        frameQueue_.clear();
        frameIndexQueue_.clear();
        transforms_.clear();
        path_.clear();
        smoothedPath_.clear();
        boxKernel_.clear();

        prevGrayCPU_.release();

    #ifdef HAVE_OPENCV_CUDAARITHM
        prevGrayGPU_.release();
        gpuFrameBuffer_.release();
        gpuGrayBuffer_.release();
        gpuPrevGrayBuffer_.release();
        gpuWarpBuffer_.release();
        
        // Synchronize streams before cleanup - CUDA streams don't have empty() method
        cudaStream_.waitForCompletion();
        analysisStream_.waitForCompletion();
        warpStream_.waitForCompletion();
    #endif

        prevKeypointsCPU_.clear();

    #ifdef HAVE_OPENCV_CUDAOPTFLOW
        prevPtsGPU_.release();
    #endif

        firstFrame_ = true;
        nextFrameIndex_ = 0;
        frameWidth_ = 0;
        frameHeight_ = 0;
        origSize_ = cv::Size();
    }

    cv::Mat Stabilizer::stabilize(const cv::Mat &frame)
    {
    static int frameTicker = 0;
    // Reduced frame skipping to prevent shaking - process more frames for stability
    constexpr int SKIP_RATE = 1;   // Process every frame for better stability
        if(frame.empty()) {
            return cv::Mat();
        }

        if(params_.cropNZoom && origSize_.empty()) {
            origSize_ = frame.size();
        }

        if(firstFrame_) {
            // Initialize with first frame - optimized for Jetson Orin Nano
            frameWidth_ = frame.cols;
            frameHeight_ = frame.rows;
            
            // Ultra-low latency analysis resolution for real-time performance
            constexpr int ANALYSIS_W = 480, ANALYSIS_H = 270;  // Smaller for speed
            
            cv::Mat firstSmall;
            if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
                // Ensure buffer is large enough for the frame
                if(gpuFrameBuffer_.rows < frame.rows || gpuFrameBuffer_.cols < frame.cols) {
                    gpuFrameBuffer_.create(std::max(frame.rows, 2160), std::max(frame.cols, 3840), CV_8UC3);
                }
                
                // Upload to pre-allocated buffer with safe ROI
                cv::cuda::GpuMat gpuFrame = gpuFrameBuffer_(cv::Rect(0, 0, frame.cols, frame.rows));
                gpuFrame.upload(frame, analysisStream_);
                
                // Ensure analysis buffer is large enough
                if(gpuGrayBuffer_.rows < ANALYSIS_H || gpuGrayBuffer_.cols < ANALYSIS_W) {
                    gpuGrayBuffer_.create(std::max(ANALYSIS_H, 1080), std::max(ANALYSIS_W, 1920), CV_8UC1);
                }
                
                // Fast resize and color conversion in one pass
                cv::cuda::GpuMat gpuSmall = gpuGrayBuffer_(cv::Rect(0, 0, ANALYSIS_W, ANALYSIS_H));
                cv::cuda::resize(gpuFrame, gpuSmall, cv::Size(ANALYSIS_W, ANALYSIS_H), 0, 0, cv::INTER_LINEAR, analysisStream_);
                cv::cuda::cvtColor(gpuSmall, prevGrayGPU_, cv::COLOR_BGR2GRAY, 0, analysisStream_);
                analysisStream_.waitForCompletion();
    #endif
            } else {
                // CPU path - simple and fast
                cv::resize(frame, firstSmall, cv::Size(ANALYSIS_W, ANALYSIS_H), 0, 0, cv::INTER_LINEAR);
                cv::cvtColor(firstSmall, prevGrayCPU_, cv::COLOR_BGR2GRAY);
            }

            // GFTT for the "prev" frame
            if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAFEATURES2D
                if(gfttDetector_) {
                    cv::cuda::GpuMat cornersGPU;
                    gfttDetector_->detect(prevGrayGPU_, cornersGPU, cv::noArray());
                    // cornersGPU often Nx1, CV_32FC2. We must reshape to 1xN:
                    if(!cornersGPU.empty()) {
                        // cornersGPU.total() = N
                        // cornersGPU.channels() = 2
                        cornersGPU = cornersGPU.reshape(2, 1); // Now 1xN

                        std::vector<cv::Point2f> cornersCPU(cornersGPU.cols);
                        cornersGPU.download(cornersCPU);

                        prevKeypointsCPU_ = cornersCPU;

    #ifdef HAVE_OPENCV_CUDAOPTFLOW
                        cv::Mat ptsMat(1, (int)cornersCPU.size(), CV_32FC2, (void*)cornersCPU.data());
                        prevPtsGPU_.upload(ptsMat);
    #endif
                    } else {
                        prevKeypointsCPU_.clear();
                        prevPtsGPU_.release();
                    }
                }
                else {
                    // fallback to CPU
                    logMessage("No cudaFeatures2d => CPU GFTT fallback", false);
                    prevGrayCPU_ = convertColorAndEnhanceCPU(frame);
                    std::vector<cv::Point2f> corners;
                    cv::goodFeaturesToTrack(prevGrayCPU_, corners,
                        params_.maxCorners, params_.qualityLevel, params_.minDistance);
                    prevKeypointsCPU_ = corners;
                }
    #else
                logMessage("No HAVE_OPENCV_CUDAFEATURES2D => CPU GFTT fallback", false);
                prevGrayCPU_ = convertColorAndEnhanceCPU(frame);
                std::vector<cv::Point2f> corners;
                cv::goodFeaturesToTrack(prevGrayCPU_, corners,
                    params_.maxCorners, params_.qualityLevel, params_.minDistance);
                prevKeypointsCPU_ = corners;
    #endif
            }
            else {
                // CPU path
                std::vector<cv::Point2f> corners;
                cv::goodFeaturesToTrack(prevGrayCPU_, corners,
                    params_.maxCorners, params_.qualityLevel, params_.minDistance,
                    cv::noArray(), params_.blockSize);
                prevKeypointsCPU_ = corners;
            }

            frameQueue_.push_back(frame.clone());
            frameIndexQueue_.push_back(0);

            firstFrame_ = false;
            nextFrameIndex_ = 1;

            return cv::Mat();
        }
	if ((frameTicker++ % SKIP_RATE) != 0) {
	    // For skipped frames, avoid cloning - just queue reference and return original
	    frameQueue_.push_back(frame);  // Remove .clone() for performance
	    frameIndexQueue_.push_back(nextFrameIndex_++);
	    return frame;              // <- forward original
	}
        // subsequent frames - only clone when necessary
        frameQueue_.push_back(frame);  // Remove .clone() for performance
        frameIndexQueue_.push_back(nextFrameIndex_);
        

        generateTransform(frame);

        // Use effective smoothing radius for better performance
        int effectiveRadius = std::max(5, std::min(params_.smoothingRadius, 35));
        if(frameIndexQueue_.size() < (size_t)effectiveRadius) {
            nextFrameIndex_++;
            return cv::Mat();
        }

        cv::Mat stabilized = applyNextSmoothTransform();
        nextFrameIndex_++;
        return stabilized;
    }

    cv::Mat Stabilizer::flush()
    {
        if(frameQueue_.empty()) {
            return cv::Mat();
        }
        return applyNextSmoothTransform();
    }

    void Stabilizer::generateTransform(const cv::Mat &currFrameBGR)
    {
    // HF: Dynamic analysis resolution for drone mode
    cv::Size analysisSize;
    if (params_.droneHighFreqMode) {
        analysisSize = calculateDroneAnalysisSize(currFrameBGR);
    } else {
        // Ultra-fast analysis resolution for Jetson Orin Nano real-time performance
        analysisSize = cv::Size(480, 270);  // Reduced for speed
    }
    
    // Convert current frame to Gray with minimal CPU-GPU transfers
    cv::Mat currGrayCPU;
    #ifdef HAVE_OPENCV_CUDAARITHM
        cv::cuda::GpuMat currGrayGPU;
    #endif

        if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
            // Ensure buffer is large enough for the current frame
            if(gpuFrameBuffer_.rows < currFrameBGR.rows || gpuFrameBuffer_.cols < currFrameBGR.cols) {
                gpuFrameBuffer_.create(std::max(currFrameBGR.rows, 2160), std::max(currFrameBGR.cols, 3840), CV_8UC3);
            }
            
            // Reuse pre-allocated buffers - zero allocation during runtime
            cv::cuda::GpuMat gpuFrame = gpuFrameBuffer_(cv::Rect(0, 0, currFrameBGR.cols, currFrameBGR.rows));
            gpuFrame.upload(currFrameBGR, analysisStream_);
            
            // Ensure analysis buffer is large enough for dynamic analysis size
            if(gpuGrayBuffer_.rows < analysisSize.height || gpuGrayBuffer_.cols < analysisSize.width) {
                gpuGrayBuffer_.create(std::max(analysisSize.height, 1080), std::max(analysisSize.width, 1920), CV_8UC1);
            }
            
            cv::cuda::GpuMat gpuSmall = gpuGrayBuffer_(cv::Rect(0, 0, analysisSize.width, analysisSize.height));
            cv::cuda::resize(gpuFrame, gpuSmall, analysisSize, 0, 0, cv::INTER_LINEAR, analysisStream_);
            cv::cuda::cvtColor(gpuSmall, currGrayGPU, cv::COLOR_BGR2GRAY, 0, analysisStream_);
            
            // HF: Apply conditional CLAHE if in drone mode and feature starved
            if (params_.droneHighFreqMode && shouldApplyConditionalCLAHE(-1)) {
                // GPU CLAHE application would go here if available
                // For now, fallback to CPU path for CLAHE
            }
            // Don't wait here - let optical flow wait when needed
    #endif
        } else {
            // CPU path - optimized with INTER_LINEAR for speed
            cv::Mat small;
            cv::resize(currFrameBGR, small, analysisSize, 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(small, currGrayCPU, cv::COLOR_BGR2GRAY);
            
            // HF: Apply conditional CLAHE if in drone mode
            if (params_.droneHighFreqMode) {
                currGrayCPU = applyConditionalCLAHE(currGrayCPU);
            }
        }
        
	analysisWidth_  = analysisSize.width;
	analysisHeight_ = analysisSize.height;

        // Jetson Orin Nano optimized optical flow with safety checks
        if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAOPTFLOW
            if(prevPtsGPU_.empty() || prevPtsGPU_.cols == 0) {
                transforms_.push_back(cv::Vec3f(0.f,0.f,0.f));
            }
            else {
                // Synchronize analysis stream before optical flow
                analysisStream_.waitForCompletion();
                
                // HF: Safety check for GPU matrices and size compatibility
                if(prevGrayGPU_.empty() || currGrayGPU.empty()) {
                    if(params_.logging) {
                        logMessage("Empty GPU gray matrices, skipping optical flow", true);
                    }
                    transforms_.push_back(cv::Vec3f(0.f,0.f,0.f));
                    return;
                }
                
                // HF: Ensure previous frame matches current analysis resolution
                if(prevGrayGPU_.size() != currGrayGPU.size()) {
                    if(params_.logging) {
                        logMessage("HF: Resizing previous frame buffer to match analysis resolution", false);
                    }
                    // Resize previous frame to match current analysis size
                    cv::cuda::GpuMat resizedPrev;
                    cv::cuda::resize(prevGrayGPU_, resizedPrev, analysisSize, 0, 0, cv::INTER_LINEAR, analysisStream_);
                    prevGrayGPU_ = resizedPrev;
                    analysisStream_.waitForCompletion();
                }
                
                cv::cuda::GpuMat currPtsGPU, statusGPU, errGPU;
                
                try {
                    // Use main CUDA stream for optical flow
                    pyrLK_->calc(
                        prevGrayGPU_,
                        currGrayGPU,
                        prevPtsGPU_,
                        currPtsGPU,
                        statusGPU,
                        errGPU,
                        cudaStream_
                    );
                } catch(const cv::Exception& e) {
                    if(params_.logging) {
                        logMessage("GPU optical flow failed: " + std::string(e.what()), true);
                    }
                    transforms_.push_back(cv::Vec3f(0.f,0.f,0.f));
                    return;
                }
                
                // Add safety checks for GPU memory operations
                std::vector<cv::Point2f> currPointsCPU;
                std::vector<uchar> statusCPU;
                
                if(!currPtsGPU.empty() && currPtsGPU.cols > 0) {
                    try {
                        currPointsCPU.resize(currPtsGPU.cols);
                        currPtsGPU.download(currPointsCPU, cudaStream_);
                    } catch(const cv::Exception& e) {
                        if(params_.logging) {
                            logMessage("GPU download failed for currPts: " + std::string(e.what()), true);
                        }
                        transforms_.push_back(cv::Vec3f(0.f,0.f,0.f));
                        return;
                    }
                }
                if(!statusGPU.empty() && statusGPU.cols > 0) {
                    try {
                        statusCPU.resize(statusGPU.cols);
                        statusGPU.download(statusCPU, cudaStream_);
                    } catch(const cv::Exception& e) {
                        if(params_.logging) {
                            logMessage("GPU download failed for status: " + std::string(e.what()), true);
                        }
                        transforms_.push_back(cv::Vec3f(0.f,0.f,0.f));
                        return;
                    }
                }
                
                // Wait for downloads to complete
                cudaStream_.waitForCompletion();

                // Ultra-fast filtering - skip outlier rejection for real-time
                std::vector<cv::Point2f> validPrev, validCurr;
                validPrev.reserve(statusCPU.size() / 2);  // Pre-allocate
                validCurr.reserve(statusCPU.size() / 2);
                
                // Add bounds checking to prevent segfault
                size_t minSize = std::min({statusCPU.size(), prevKeypointsCPU_.size(), currPointsCPU.size()});
                for(size_t i=0; i<minSize; i++){
                    if(i < statusCPU.size() && statusCPU[i] && 
                       i < prevKeypointsCPU_.size() && i < currPointsCPU.size()) {
                        validPrev.push_back(prevKeypointsCPU_[i]);
                        validCurr.push_back(currPointsCPU[i]);
                    }
                }
                
                // Fast affine estimation with safety checks
                cv::Mat T = cv::Mat::eye(2,3,CV_32F);
                if(validPrev.size() >= 4 && validCurr.size() >= 4 && validPrev.size() == validCurr.size()) {
                    try {
                        cv::Mat affine = cv::estimateAffinePartial2D(validPrev, validCurr, 
                                                                   cv::noArray(), cv::RANSAC, 
                                                                   5.0, 500);  // Reduced iterations
                        if(!affine.empty() && affine.rows == 2 && affine.cols == 3) {
                            affine.convertTo(T, CV_32F);
                        }
                    } catch(const cv::Exception& e) {
                        if(params_.logging) {
                            logMessage("Affine estimation failed: " + std::string(e.what()), true);
                        }
                        // T remains as identity matrix
                    }
                }
                float dx = T.at<float>(0,2);
                float dy = T.at<float>(1,2);
                float da = std::atan2(T.at<float>(1,0), T.at<float>(0,0));
                
                // HF: Apply drone high-frequency vibration suppression
                Transform rawTransform(dx, dy, da);
                if (params_.droneHighFreqMode) {
                    rawTransform = applyDeadZoneFreeze(rawTransform);
                    rawTransform = applyMicroShakeSuppression(rawTransform);
                    rawTransform = applyRotationLowPass(rawTransform);
                    updateTranslationHistory(cv::Vec2f(rawTransform.dx, rawTransform.dy));
                }
                
                transforms_.push_back(cv::Vec3f(rawTransform.dx, rawTransform.dy, rawTransform.da));
            }
    #endif
        }
        else {
            // CPU Optical flow - optimized for speed with safety checks
            if(!prevKeypointsCPU_.empty() && !prevGrayCPU_.empty() && !currGrayCPU.empty()) {
                // HF: Ensure previous frame matches current analysis resolution
                if(prevGrayCPU_.size() != currGrayCPU.size()) {
                    if(params_.logging) {
                        logMessage("HF: Resizing previous CPU frame buffer to match analysis resolution", false);
                    }
                    cv::resize(prevGrayCPU_, prevGrayCPU_, analysisSize, 0, 0, cv::INTER_LINEAR);
                }
                
                std::vector<cv::Point2f> tmpCurr;
                std::vector<uchar> status;
                std::vector<float> err;
                
                try {
                    // Optimized LK parameters for speed
                    cv::calcOpticalFlowPyrLK(
                        prevGrayCPU_, currGrayCPU,
                        prevKeypointsCPU_,
                        tmpCurr,
                        status, err,
                        cv::Size(15,15),  // Reduced window size
                        2,                // Reduced pyramid levels
                        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 20, 0.03)
                    );
                } catch(const cv::Exception& e) {
                    if(params_.logging) {
                        logMessage("CPU optical flow failed: " + std::string(e.what()), true);
                    }
                    transforms_.push_back(cv::Vec3f(0.f,0.f,0.f));
                    return;
                }
                
                // Fast filtering - pre-allocate for speed
                std::vector<cv::Point2f> validPrev, validCurr;
                validPrev.reserve(status.size() / 2);
                validCurr.reserve(status.size() / 2);
                
                // Add bounds checking to prevent segfault
                size_t minSize = std::min({status.size(), prevKeypointsCPU_.size(), tmpCurr.size()});
                for(size_t i=0; i<minSize; i++) {
                    if(i < status.size() && status[i] && 
                       i < prevKeypointsCPU_.size() && i < tmpCurr.size()) {
                        validPrev.push_back(prevKeypointsCPU_[i]);
                        validCurr.push_back(tmpCurr[i]);
                    }
                }
                
                // Fast affine estimation with safety checks
                cv::Mat T = cv::Mat::eye(2,3,CV_32F);
                if(validPrev.size() >= 4 && validCurr.size() >= 4 && validPrev.size() == validCurr.size()) {
                    try {
                        cv::Mat affine = cv::estimateAffinePartial2D(validPrev, validCurr, 
                                                                   cv::noArray(), cv::RANSAC, 
                                                                   5.0, 500);
                        if(!affine.empty() && affine.rows == 2 && affine.cols == 3) {
                            affine.convertTo(T, CV_32F);
                        }
                    } catch(const cv::Exception& e) {
                        if(params_.logging) {
                            logMessage("CPU affine estimation failed: " + std::string(e.what()), true);
                        }
                        // T remains as identity matrix
                    }
                }
                float dx = T.at<float>(0,2);
                float dy = T.at<float>(1,2);
                float da = std::atan2(T.at<float>(1,0), T.at<float>(0,0));
                
                // HF: Apply drone high-frequency vibration suppression
                Transform rawTransform(dx, dy, da);
                if (params_.droneHighFreqMode) {
                    rawTransform = applyDeadZoneFreeze(rawTransform);
                    rawTransform = applyMicroShakeSuppression(rawTransform);
                    rawTransform = applyRotationLowPass(rawTransform);
                    updateTranslationHistory(cv::Vec2f(rawTransform.dx, rawTransform.dy));
                }
                
                transforms_.push_back(cv::Vec3f(rawTransform.dx, rawTransform.dy, rawTransform.da));
            }
            else {
                transforms_.push_back(cv::Vec3f(0.f,0.f,0.f));
            }
        }

        // Path
        if(path_.empty()) {
            path_.push_back(transforms_.back());
        } else {
            cv::Vec3f last = path_.back();
            cv::Vec3f now = transforms_.back();
            path_.push_back(cv::Vec3f(last[0]+now[0], last[1]+now[1], last[2]+now[2]));
        }
        smoothedPath_ = path_;
        
        // Skip adaptive parameters for performance - disabled by default in config
        if (params_.adaptiveSmoothing) {
            updateAdaptiveParameters();
        }

        // Optimized feature detection for next iteration - reduced frequency for speed
        static int featureDetectionCounter = 0;
        if((++featureDetectionCounter % 2) == 0) {  // Detect features every 2nd frame for speed
            if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
                cv::cuda::GpuMat currGray = currGrayGPU;
    #ifdef HAVE_OPENCV_CUDAFEATURES2D
                if(gfttDetector_) {
                    cv::cuda::GpuMat cornersGPU;
                    
                    // Use analysis stream for feature detection
                    gfttDetector_->detect(currGray, cornersGPU, cv::noArray(), analysisStream_);
                    
                    if(!cornersGPU.empty()) {
                        cornersGPU = cornersGPU.reshape(2, 1);  // Reshape to 1xN
                        
                        std::vector<cv::Point2f> cornersCPU(cornersGPU.cols);
                        cornersGPU.download(cornersCPU, analysisStream_);
                        
                        // Wait for download to complete
                        analysisStream_.waitForCompletion();
                        
                        prevKeypointsCPU_ = cornersCPU;
    #ifdef HAVE_OPENCV_CUDAOPTFLOW
                        if(!cornersCPU.empty()) {
                            cv::Mat ptsMat(1, (int)cornersCPU.size(), CV_32FC2, (void*)cornersCPU.data());
                            prevPtsGPU_.upload(ptsMat, analysisStream_);
                        } else {
                            prevPtsGPU_.release();
                        }
    #endif
                    } else {
                        prevKeypointsCPU_.clear();
                        prevPtsGPU_.release();
                    }
                }
    #endif
    #endif
            }
            else {
                // Fast CPU feature detection - reduced parameters
                std::vector<cv::Point2f> corners;
                cv::goodFeaturesToTrack(currGrayCPU, corners,
                    std::min(params_.maxCorners, 200),  // Limit features
                    0.02,     // Lower quality threshold for speed
                    15.0,     // Larger min distance
                    cv::noArray(), 3);  // Smaller block size
                prevKeypointsCPU_ = corners;
            }
        }

        // Update prevGray
        if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
            if(!currGrayGPU.empty()) {
                prevGrayGPU_ = currGrayGPU;
            }
    #endif
        } else {
            if(!currGrayCPU.empty()) {
                prevGrayCPU_ = currGrayCPU;
            }
        }
    }

    cv::Mat Stabilizer::applyNextSmoothTransform()
    {
        if(frameQueue_.empty()) {
            return cv::Mat();
        }
        cv::Mat oldestFrame = frameQueue_.front();
        int oldestIdx = frameIndexQueue_.front();
        frameQueue_.pop_front();
        frameIndexQueue_.pop_front();

        // Add bounds checking for oldestIdx to prevent segfault
        if((size_t)oldestIdx >= transforms_.size() || (size_t)oldestIdx >= path_.size() || 
           (size_t)oldestIdx >= smoothedPath_.size()) {
            if(params_.logging) {
                logMessage("Index out of bounds in applyNextSmoothTransform, returning original frame", true);
            }
            return oldestFrame;
        }

        // Extract path components
        std::vector<float> px, py, pa;
        px.reserve(path_.size());
        py.reserve(path_.size());
        pa.reserve(path_.size());
        for(const auto &v : path_) {
            px.push_back(v[0]);
            py.push_back(v[1]);
            pa.push_back(v[2]);
        }
        
        // Professional multi-frame smoothing (iPhone/GoPro-like)
        std::vector<float> sx, sy, sa;
        
        // Enhanced smoothing with motion-aware algorithms
        if(params_.smoothingMethod == "gaussian") {
            // Gaussian smoothing for high-quality results
            sx = gaussianFilterConvolve(px, params_.gaussianSigma);
            sy = gaussianFilterConvolve(py, params_.gaussianSigma);
            sa = gaussianFilterConvolve(pa, params_.gaussianSigma);
        } else if(params_.smoothingMethod == "kalman") {
            // Kalman filter for predictive smoothing
            sx = kalmanFilterSmooth(px);
            sy = kalmanFilterSmooth(py);
            sa = kalmanFilterSmooth(pa);
        } else {
            // Enhanced box filter with adaptive radius
            int adaptiveRadius = calculateAdaptiveRadius(px, py, pa);
            
            // Temporarily update radius for adaptive smoothing
            int originalRadius = params_.smoothingRadius;
            params_.smoothingRadius = adaptiveRadius;
            boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);
            
            sx = boxFilterConvolve(px);
            sy = boxFilterConvolve(py);
            sa = boxFilterConvolve(pa);
            
            // Restore original radius
            params_.smoothingRadius = originalRadius;
            boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);
        }

        // Add safety checks for smoothedPath size to prevent segfault
        if(smoothedPath_.size() != path_.size() || smoothedPath_.size() <= (size_t)oldestIdx) {
            if(params_.logging) {
                logMessage("SmoothedPath size mismatch, rebuilding...", true);
            }
            smoothedPath_.resize(path_.size());
            for(size_t i=0; i<path_.size() && i<sx.size() && i<sy.size() && i<sa.size(); i++) {
                smoothedPath_[i] = cv::Vec3f(sx[i], sy[i], sa[i]);
            }
        }

        smoothedPath_.resize(path_.size());
        for(size_t i=0; i<path_.size() && i<sx.size() && i<sy.size() && i<sa.size(); i++) {
            smoothedPath_[i] = cv::Vec3f(sx[i], sy[i], sa[i]);
        }

        // Professional motion analysis and prediction with bounds checking
        if((size_t)oldestIdx >= transforms_.size() || (size_t)oldestIdx >= path_.size() || 
           (size_t)oldestIdx >= smoothedPath_.size()) {
            if(params_.logging) {
                logMessage("Invalid index for motion analysis, using safe fallback", true);
            }
            return oldestFrame;
        }
        
        cv::Vec3f raw = transforms_[oldestIdx];
        cv::Vec3f diff = smoothedPath_[oldestIdx] - path_[oldestIdx];
        
        // Advanced motion intent detection (like iPhone Action mode)
        if (oldestIdx > 0) {
            MotionIntent intent = analyzeMotionIntent(raw, oldestIdx);
            float adaptiveStrength = calculateAdaptiveStabilizationStrength(intent, raw);
            
            switch(intent) {
                case MotionIntent::DELIBERATE_PAN:
                    // Minimal correction for intentional camera movement
                    diff *= 0.5f;
                    if (params_.logging) {
                        logMessage("Deliberate panning detected - preserving motion", false);
                    }
                    break;
                    
                case MotionIntent::SHAKE_REMOVAL:
                    // Strong correction for camera shake
                    diff *= 1.0f;
                    if (params_.logging) {
                        logMessage("Camera shake detected - applying strong stabilization", false);
                    }
                    break;
                    
                case MotionIntent::FOLLOW_ACTION:
                    // Balanced correction for action following
                    diff *= 0.8f;
                    if (params_.logging) {
                        logMessage("Action following detected - balanced stabilization", false);
                    }
                    break;
                    
                default:
                    // Adaptive correction based on motion quality
                    diff *= adaptiveStrength;
                    break;
            }
        }
        
        cv::Vec3f tSmooth = raw + diff;

        float dx = tSmooth[0];
        float dy = tSmooth[1];
        float da = tSmooth[2];
        
        // Lock horizon if enabled
        if (params_.horizonLock) {
            da = 0.0f;
        }

        // 2x3 transformation matrix
        cv::Mat T(2,3,CV_32F);
        T.at<float>(0,0) =  std::cos(da);
        T.at<float>(0,1) = -std::sin(da);
        T.at<float>(1,0) =  std::sin(da);
        T.at<float>(1,1) =  std::cos(da);
        T.at<float>(0,2) =  dx;
        T.at<float>(1,2) =  dy;

        // Handle different border types
        cv::Mat frameWithBorder;
        
        // Check if we're using the special "fade" border type
        if (params_.borderType == "fade") {
            if (params_.borderSize > 0 && !params_.cropNZoom) {
                // For fade border, we need to create initial background if it doesn't exist
                if (borderHistory_.empty()) {
                    // Initialize with black borders
                    cv::copyMakeBorder(
                        oldestFrame, borderHistory_,
                        params_.borderSize, params_.borderSize,
                        params_.borderSize, params_.borderSize,
                        cv::BORDER_CONSTANT, cv::Scalar(0,0,0)
                    );
                    fadeFrameCount_ = 0;
                }
                
                // Create current frame with border
                cv::copyMakeBorder(
                    oldestFrame, frameWithBorder,
                    params_.borderSize, params_.borderSize,
                    params_.borderSize, params_.borderSize,
                    cv::BORDER_CONSTANT, cv::Scalar(0,0,0)
                );
                
                // Apply the fading effect - only to border areas
                // First, create a mask for border regions
                cv::Mat borderMask = cv::Mat::zeros(frameWithBorder.size(), CV_8UC1);
                // Set the inner region to 0 (frame) and outer region to 255 (border)
                cv::rectangle(
                    borderMask, 
                    cv::Rect(params_.borderSize, params_.borderSize, 
                             oldestFrame.cols, oldestFrame.rows),
                    cv::Scalar(0), -1  // -1 means filled
                );
                cv::rectangle(
                    borderMask,
                    cv::Rect(0, 0, frameWithBorder.cols, frameWithBorder.rows),
                    cv::Scalar(255), -1
                );
                
                // Now blend the current frame with border and the history where the mask is active
                float alpha = params_.fadeAlpha;
                cv::Mat blended;
                
                // Use different alpha based on frame counter for gradual fade-in
                if (fadeFrameCount_ < params_.fadeDuration) {
                    // Gradually fade in the history
                    alpha = alpha * (static_cast<float>(fadeFrameCount_) / params_.fadeDuration);
                    fadeFrameCount_++;
                }
                
cv::Mat blendedBorder;
cv::addWeighted(borderHistory_,        // src1
                alpha,                 // weight for src1
                frameWithBorder,       // src2
                1.0f - alpha,          // weight for src2
                0.0,                   // gamma
                blendedBorder);        // dst

// 2) Copy the blended result **only** where we have a border
blendedBorder.copyTo(frameWithBorder, borderMask);
                
                // Now update the border history with the current stabilized frame
                // We'll do this after warping
            } else {
                frameWithBorder = oldestFrame;
            }
        } else {
            // Standard OpenCV border types
            if (params_.borderSize > 0 && !params_.cropNZoom) {
                cv::copyMakeBorder(
                    oldestFrame, frameWithBorder,
                    params_.borderSize, params_.borderSize,
                    params_.borderSize, params_.borderSize,
                    borderMode_, cv::Scalar(0,0,0)
                );
            } else {
                frameWithBorder = oldestFrame;
            }
        }

        // Jetson Orin Nano optimized warping with hardware acceleration and safety checks
        cv::Mat stabilized;
        if(useGpu_) {
    #if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
            try {
                // Safety check for input frame
                if(frameWithBorder.empty()) {
                    if(params_.logging) {
                        logMessage("Empty frame for GPU warping, returning original", true);
                    }
                    return oldestFrame;
                }
                
                // Ensure warp buffer is large enough
                if(gpuWarpBuffer_.rows < frameWithBorder.rows || gpuWarpBuffer_.cols < frameWithBorder.cols) {
                    gpuWarpBuffer_.create(std::max(frameWithBorder.rows, 2160), std::max(frameWithBorder.cols, 3840), CV_8UC3);
                }
                
                // Ensure frame buffer is large enough
                if(gpuFrameBuffer_.rows < frameWithBorder.rows || gpuFrameBuffer_.cols < frameWithBorder.cols) {
                    gpuFrameBuffer_.create(std::max(frameWithBorder.rows, 2160), std::max(frameWithBorder.cols, 3840), CV_8UC3);
                }
                
                // Use pre-allocated GPU buffers for zero-copy operation
                cv::cuda::GpuMat gpuIn = gpuWarpBuffer_(cv::Rect(0, 0, frameWithBorder.cols, frameWithBorder.rows));
                cv::cuda::GpuMat gpuOut = gpuFrameBuffer_(cv::Rect(0, 0, frameWithBorder.cols, frameWithBorder.rows));
                
                // Upload using warp stream for parallel processing
                gpuIn.upload(frameWithBorder, warpStream_);
                
                // Hardware-accelerated warping
                cv::cuda::warpAffine(
                    gpuIn, gpuOut, T, gpuIn.size(),
                    cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), warpStream_
                );
                
                // Download result
                gpuOut.download(stabilized, warpStream_);
                warpStream_.waitForCompletion();
            } catch(const cv::Exception& e) {
                if(params_.logging) {
                    logMessage("GPU warping failed: " + std::string(e.what()) + ", falling back to CPU", true);
                }
                // Fallback to CPU warping
                cv::warpAffine(frameWithBorder, stabilized, T, frameWithBorder.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            }
    #else
            // CPU fallback with optimized parameters
            cv::warpAffine(
                frameWithBorder, stabilized,
                T, frameWithBorder.size(),
                cv::INTER_LINEAR, cv::BORDER_CONSTANT
            );
    #endif
        } else {
            // Optimized CPU warping with safety checks
            try {
                if(frameWithBorder.empty()) {
                    if(params_.logging) {
                        logMessage("Empty frame for CPU warping, returning original", true);
                    }
                    return oldestFrame;
                }
                cv::warpAffine(
                    frameWithBorder, stabilized,
                    T, frameWithBorder.size(),
                    cv::INTER_LINEAR, cv::BORDER_CONSTANT
                );
            } catch(const cv::Exception& e) {
                if(params_.logging) {
                    logMessage("CPU warping failed: " + std::string(e.what()) + ", returning original frame", true);
                }
                return oldestFrame;
            }
        }
        
        // Update the border history for fade effect
        if (params_.borderType == "fade" && params_.borderSize > 0) {
            // Use the current stabilized frame to update the history
            if (!borderHistory_.empty() && borderHistory_.size() == stabilized.size()) {
                // Create a mask for border regions again
                cv::Mat borderMask = cv::Mat::zeros(stabilized.size(), CV_8UC1);
                cv::rectangle(
                    borderMask, 
                    cv::Rect(params_.borderSize, params_.borderSize, 
                             oldestFrame.cols, oldestFrame.rows),
                    cv::Scalar(0), -1
                );
                cv::rectangle(
                    borderMask,
                    cv::Rect(0, 0, stabilized.cols, stabilized.rows),
                    cv::Scalar(255), -1
                );
                
                // Only update the border regions in the history
                for (int y = 0; y < borderMask.rows; y++) {
                    for (int x = 0; x < borderMask.cols; x++) {
                        if (borderMask.at<uchar>(y, x) > 0) {  // If this is a border pixel
                            for (int c = 0; c < 3; c++) {  // For each color channel
                                // Update history with current stabilized frame
                                // Use a slow decay to gradually incorporate new content
                                float updateRate = 0.1f;
                                borderHistory_.at<cv::Vec3b>(y, x)[c] = 
                                    static_cast<uchar>((1.0f - updateRate) * borderHistory_.at<cv::Vec3b>(y, x)[c] + 
                                                   updateRate * stabilized.at<cv::Vec3b>(y, x)[c]);
                            }
                        }
                    }
                }
            } else {
                // If the history is empty or size mismatch, initialize it
                borderHistory_ = stabilized.clone();
            }
        }

        // Crop and zoom if enabled
if (params_.cropNZoom && params_.borderSize > 0)
{
    int b = params_.borderSize;
    int w = stabilized.cols - 2*b;
    int h = stabilized.rows - 2*b;

    if (w <= 0 || h <= 0)            // border larger than image
        return stabilized;           // skip cropping this frame

    cv::Rect roi(b, b, w, h);
    cv::Mat cropped = stabilized(roi).clone();

    if (!origSize_.empty() && !cropped.empty())
        cv::resize(cropped, cropped, origSize_);

    return cropped;
}
        else if(params_.cropNZoom && params_.borderSize==0) {
            return stabilized;
        }

        // Apply Virtual Canvas Stabilization if enabled
        if (params_.enableVirtualCanvas) {
            cv::Vec3f currentTransform(dx, dy, da);
            updateTemporalFrameBuffer(oldestFrame, currentTransform);
            stabilized = applyVirtualCanvasStabilization(oldestFrame, currentTransform);
        }

        return stabilized;
    }

std::vector<float> Stabilizer::boxFilterConvolve(const std::vector<float> &path)
{
    if(path.empty()) return {};
    
    // HF: More aggressive smoothing for drone mode freeze shot
    int r = params_.droneHighFreqMode ? 
            std::max(10, std::min(params_.smoothingRadius, 50)) :  // Larger radius for drone mode
            std::max(2, std::min(params_.smoothingRadius, 8));     // Smaller radius for normal mode

    if(path.size() <= static_cast<size_t>(r)) {
        return path;  // Return original if too small
    }
    
    // Enhanced smoothing for steady shot
    std::vector<float> result(path.size());
    
    // More aggressive box filter for freeze shot capability
    for(size_t i = 0; i < path.size(); i++) {
        float sum = 0.0f;
        int count = 0;
        
        int start = std::max(0, static_cast<int>(i) - r);
        int end = std::min(static_cast<int>(path.size()) - 1, static_cast<int>(i) + r);
        
        for(int j = start; j <= end; j++) {
            sum += path[j];
            count++;
        }
        
        result[i] = sum / count;
    }
    
    return result;
}

    // Ultra-fast color conversion for Jetson Orin Nano - no enhancement
    cv::Mat Stabilizer::convertColorAndEnhanceCPU(const cv::Mat &frameBGR)
    {
        cv::Mat gray;
        cv::cvtColor(frameBGR, gray, cv::COLOR_BGR2GRAY);
        return gray;  // Skip CLAHE for real-time performance
    }

    #ifdef HAVE_OPENCV_CUDAARITHM
    // Ultra-fast GPU color conversion - no enhancement
    cv::cuda::GpuMat Stabilizer::convertColorAndEnhanceGPU(const cv::Mat &frameBGR)
    {
        cv::cuda::GpuMat gpuIn, gpuGray;
        gpuIn.upload(frameBGR);
        cv::cuda::cvtColor(gpuIn, gpuGray, cv::COLOR_BGR2GRAY);
        return gpuGray;  // Skip CLAHE for real-time performance
    }
    #endif

    // Implements feature detection based on selected method
    std::vector<cv::Point2f> Stabilizer::detectFeatures(const cv::Mat &grayFrame)
    {
        std::vector<cv::Point2f> corners;
        cv::Rect roi = params_.useROI ? calculateROI(grayFrame) : cv::Rect(0, 0, grayFrame.cols, grayFrame.rows);
        
        // Apply ROI if specified
        cv::Mat roiFrame = grayFrame(roi);
        
        switch (params_.featureDetector) {
            case Parameters::GFTT:
                cv::goodFeaturesToTrack(
                    roiFrame, corners,
                    params_.maxCorners,
                    params_.qualityLevel,
                    params_.minDistance,
                    cv::noArray(),
                    params_.blockSize
                );
                break;
                
            case Parameters::ORB: {
                // Use ORB detector
                cv::Ptr<cv::ORB> detector = cv::ORB::create(params_.orbFeatures);
                std::vector<cv::KeyPoint> keypoints;
                detector->detect(roiFrame, keypoints);
                
                // Convert KeyPoint to Point2f
                corners.reserve(keypoints.size());
                for (const auto& kp : keypoints) {
                    corners.push_back(kp.pt);
                }
                break;
            }
                
            case Parameters::FAST: {
                // Use FAST detector
                std::vector<cv::KeyPoint> keypoints;
                cv::FAST(roiFrame, keypoints, params_.fastThreshold);
                
                // Convert KeyPoint to Point2f
                corners.reserve(keypoints.size());
                for (const auto& kp : keypoints) {
                    corners.push_back(kp.pt);
                }
                break;
            }
                
            case Parameters::BRISK: {
                // Use BRISK detector
                cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
                std::vector<cv::KeyPoint> keypoints;
                detector->detect(roiFrame, keypoints);
                
                // Convert KeyPoint to Point2f
                corners.reserve(keypoints.size());
                for (const auto& kp : keypoints) {
                    corners.push_back(kp.pt);
                }
                break;
            }
        }
        
        // If using ROI, adjust points to global coordinates
        if (params_.useROI) {
            for (auto& point : corners) {
                point.x += roi.x;
                point.y += roi.y;
            }
        }
        
        return corners;
    }

#ifdef HAVE_OPENCV_CUDAFEATURES2D
    cv::cuda::GpuMat Stabilizer::detectFeaturesGPU(const cv::cuda::GpuMat &grayFrameGPU)
    {
        cv::cuda::GpuMat cornersGPU;
        cv::Rect roi = params_.useROI ? calculateROI(cv::Mat()) : cv::Rect(0, 0, grayFrameGPU.cols, grayFrameGPU.rows);
        
          roi &= cv::Rect(0, 0, grayFrameGPU.cols, grayFrameGPU.rows);
    if (roi.empty())                                     // safety net
        roi = cv::Rect(0, 0, grayFrameGPU.cols, grayFrameGPU.rows);
        
        // Apply ROI if specified
        cv::cuda::GpuMat roiFrame;
        if (params_.useROI) {
            roiFrame = grayFrameGPU(roi);
        } else {
            roiFrame = grayFrameGPU;
        }
        
        // For now, we only support GFTT for GPU
        // Enhanced implementations could add more GPU feature detectors
        gfttDetector_->detect(roiFrame, cornersGPU, cv::noArray());
        
        // Reshape to 1xN format for optical flow
        if (!cornersGPU.empty()) {
            cornersGPU = cornersGPU.reshape(2, 1);
            
            // If using ROI, need to adjust points (download, shift, upload)
            if (params_.useROI) {
                std::vector<cv::Point2f> cornersCPU(cornersGPU.cols);
                cornersGPU.download(cornersCPU);
                
                for (auto& point : cornersCPU) {
                    point.x += roi.x;
                    point.y += roi.y;
                }
                
                cv::Mat ptsMat(1, (int)cornersCPU.size(), CV_32FC2, (void*)cornersCPU.data());
                cornersGPU.upload(ptsMat);
            }
        }
        
        return cornersGPU;
    }
#endif

    // Filter outliers from matched points
    void Stabilizer::filterOutliers(std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &currPts)
    {
        if (prevPts.size() != currPts.size() || prevPts.empty()) {
            return;
        }
        
        // Calculate motion vectors
        std::vector<cv::Point2f> motions(prevPts.size());
        for (size_t i = 0; i < prevPts.size(); i++) {
            motions[i] = currPts[i] - prevPts[i];
        }
        
        // Calculate mean motion
        cv::Point2f meanMotion(0, 0);
        for (const auto& m : motions) {
            meanMotion += m;
        }
        meanMotion.x /= motions.size();
        meanMotion.y /= motions.size();
        
        // Calculate standard deviation
        float stdX = 0, stdY = 0;
        for (const auto& m : motions) {
            stdX += (m.x - meanMotion.x) * (m.x - meanMotion.x);
            stdY += (m.y - meanMotion.y) * (m.y - meanMotion.y);
        }
        stdX = std::sqrt(stdX / motions.size());
        stdY = std::sqrt(stdY / motions.size());
        
        // Threshold for outlier rejection (3 standard deviations by default)
        float threshX = params_.outlierThreshold * stdX;
        float threshY = params_.outlierThreshold * stdY;
        
        // Filter points
        std::vector<cv::Point2f> filteredPrev, filteredCurr;
        for (size_t i = 0; i < motions.size(); i++) {
            if (std::abs(motions[i].x - meanMotion.x) <= threshX && 
                std::abs(motions[i].y - meanMotion.y) <= threshY) {
                filteredPrev.push_back(prevPts[i]);
                filteredCurr.push_back(currPts[i]);
            }
        }
        
        // Only update if we have enough points left
        if (filteredPrev.size() > 10) {
            prevPts = filteredPrev;
            currPts = filteredCurr;
        }
    }

    // Gaussian smoothing of path
    std::vector<float> Stabilizer::gaussianFilterConvolve(const std::vector<float> &path, float sigma)
    {
        if (path.empty()) return {};
        
        // Determine kernel size (6*sigma should contain >99% of distribution)
        int kernelSize = std::max(3, static_cast<int>(std::ceil(6 * sigma)));
        if (kernelSize % 2 == 0) kernelSize++; // Make sure it's odd
        
        // Create Gaussian kernel
        std::vector<float> kernel(kernelSize);
        float sum = 0.0f;
        int center = kernelSize / 2;
        
        for (int i = 0; i < kernelSize; i++) {
            float x = static_cast<float>(i - center);
            kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
            sum += kernel[i];
        }
        
        // Normalize kernel
        for (float &k : kernel) {
            k /= sum;
        }
        
        // Apply kernel
        std::vector<float> result(path.size());
        
        // Pad with reflected values
        std::vector<float> padded(path.size() + 2 * center);
        for (size_t i = 0; i < center; i++) {
            padded[i] = path[center - i];
        }
        for (size_t i = 0; i < path.size(); i++) {
            padded[center + i] = path[i];
        }
        for (size_t i = 0; i < center; i++) {
            padded[center + path.size() + i] = path[path.size() - 1 - i];
        }
        
        // Convolve
        for (size_t i = 0; i < path.size(); i++) {
            float sum = 0.0f;
            for (int j = 0; j < kernelSize; j++) {
                sum += padded[i + j] * kernel[j];
            }
            result[i] = sum;
        }
        
        return result;
    }

    // Kalman filter-based smoothing
    std::vector<float> Stabilizer::kalmanFilterSmooth(const std::vector<float> &path)
    {
        if (path.empty()) return {};
        
        // Create Kalman filter
        cv::KalmanFilter kf(2, 1, 0);
        
        // State transition matrix (x_t = x_t-1 + v_t-1, v_t = v_t-1)
        kf.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
        
        // Measurement matrix (measure only position)
        cv::Mat measurement = cv::Mat::zeros(1, 1, CV_32F);
        kf.measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);
        
        // Process noise (use small values)
        kf.processNoiseCov = (cv::Mat_<float>(2, 2) << 0.01, 0, 0, 0.01);
        
        // Measurement noise (tune this parameter)
        kf.measurementNoiseCov = (cv::Mat_<float>(1, 1) << 0.1);
        
        // Initialize with first point
        kf.statePost = (cv::Mat_<float>(2, 1) << path[0], 0);
        
        std::vector<float> result(path.size());
        result[0] = path[0];
        
        // Forward pass (filtering)
        for (size_t i = 1; i < path.size(); i++) {
            // Predict
            cv::Mat prediction = kf.predict();
            
            // Update measurement
            measurement.at<float>(0) = path[i];
            
            // Correct
            cv::Mat estimated = kf.correct(measurement);
            
            // Save filtered result
            result[i] = estimated.at<float>(0);
        }
        
        return result;
    }

    // Adaptive smoothing based on motion
    void Stabilizer::adaptSmoothingRadius(const cv::Vec3f &recentMotion)
    {
        // Calculate motion magnitude (translation only, rotation can be added if needed)
        float magnitude = std::sqrt(recentMotion[0] * recentMotion[0] + recentMotion[1] * recentMotion[1]);
        
        // Map motion magnitude to smoothing radius
        // Low motion -> high smoothing radius (more stable)
        // High motion -> low smoothing radius (more responsive)
        
        // Simple linear mapping between min and max
        // This could be replaced with a more sophisticated function
        float motionScale = std::max(0.0f, std::min(1.0f, magnitude / 50.0f)); // Normalize to [0,1]
        
        // Inverse mapping: high motion = low radius, low motion = high radius
        motionScale = 1.0f - motionScale;
        
        // Map to radius range
        int newRadius = params_.minSmoothingRadius + 
                        static_cast<int>(motionScale * (params_.maxSmoothingRadius - params_.minSmoothingRadius));
        
        // Update radius if it changed
        if (newRadius != params_.smoothingRadius) {
            params_.smoothingRadius = newRadius;
            
            // Update box filter kernel
            boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);
            
            if (params_.logging) {
                logMessage("Adaptive smoothing radius adjusted to: " + std::to_string(params_.smoothingRadius));
            }
        }
    }

    // Determine if motion is intentional based on magnitude and consistency
    bool Stabilizer::isIntentionalMotion(const cv::Vec3f &motion)
    {
        // Calculate motion magnitude 
        float magnitude = std::sqrt(motion[0] * motion[0] + motion[1] * motion[1]);
        
        // If magnitude is large, likely intentional
        if (magnitude > params_.intentionalMotionThreshold) {
            return true;
        }
        
        // Additional logic could be added here to check for consistent motion direction
        return false;
    }

    // Predict next motion based on previous motion patterns
    cv::Vec3f Stabilizer::predictNextMotion()
    {
        // Simple prediction: extrapolate from recent transforms
        if (transforms_.size() < 3) {
            // Not enough data, return zero motion
            return cv::Vec3f(0.0f, 0.0f, 0.0f);
        }
        
        // Linear extrapolation based on last two transforms
        cv::Vec3f lastMotion = transforms_[transforms_.size() - 1];
        cv::Vec3f prevMotion = transforms_[transforms_.size() - 2];
        cv::Vec3f delta = lastMotion - prevMotion;
        
        // Predicted = last + delta
        return lastMotion + delta;
    }

    // Calculate Region of Interest for feature detection
    cv::Rect Stabilizer::calculateROI(const cv::Mat &frame)
    {
        // If user-specified ROI is valid, use it
        if (params_.roi.width > 0 && params_.roi.height > 0) {
            return params_.roi;
        }
        
        // Calculate a default ROI (center 60% of the frame)
        int width, height;
        
if (!frame.empty()) {
    width  = frame.cols;          // caller passed an image
    height = frame.rows;
}
else if (analysisWidth_ > 0 && analysisHeight_ > 0) {
    // NEW: when no frame was supplied (GPU path) use analysis size
    width  = analysisWidth_;
    height = analysisHeight_;
}
else {
    // final fallback – original full‑res frame size
    width  = frameWidth_;
    height = frameHeight_;
}
        
        int x = width / 5;
        int y = height / 5;
        int w = width * 3 / 5;
        int h = height * 3 / 5;
        
        return cv::Rect(x, y, w, h);
    }

    // Update adaptive parameters based on motion analysis
    void Stabilizer::updateAdaptiveParameters()
    {
        // Skip if not enough data or adaptive features not enabled
        if (transforms_.size() < 3 || !params_.adaptiveSmoothing) {
            return;
        }
        
        // Get latest motion
        cv::Vec3f recentMotion = transforms_.back();
        
        // Update smoothing radius based on motion magnitude
        adaptSmoothingRadius(recentMotion);
    }

    // Apply multi-stage smoothing pipeline (VT3000 style)
    void Stabilizer::applyMultiStageSmoothing(std::vector<float> &x, std::vector<float> &y, std::vector<float> &a)
    {
        if (x.empty() || y.empty() || a.empty()) {
            return;
        }
        
        // Stage 1: Initial smoothing with stageOneRadius
        std::vector<float> x1, y1, a1;
        
        if (params_.smoothingMethod == "gaussian") {
            x1 = gaussianFilterConvolve(x, params_.gaussianSigma);
            y1 = gaussianFilterConvolve(y, params_.gaussianSigma);
            a1 = gaussianFilterConvolve(a, params_.gaussianSigma);
        } 
        else if (params_.smoothingMethod == "kalman") {
            x1 = kalmanFilterSmooth(x);
            y1 = kalmanFilterSmooth(y);
            a1 = kalmanFilterSmooth(a);
        }
        else {
            // Default to box filter
            // Save original smoothing radius
            int originalRadius = params_.smoothingRadius;
            
            // Use stage one radius for first pass
            params_.smoothingRadius = params_.stageOneRadius;
            boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);
            
            x1 = boxFilterConvolve(x);
            y1 = boxFilterConvolve(y);
            a1 = boxFilterConvolve(a);
            
            // Restore original radius
            params_.smoothingRadius = originalRadius;
            boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);
        }
        
        // Stage 2: Filter by frequency characteristics if enabled
        if (params_.jitterFrequency != Parameters::ADAPTIVE) {
            // HF: Use improved jitter frequency mapping
            double cutoffFreq = mapJitterFrequencyToCutoff(params_.jitterFrequency);
            
            x1 = butterworthFilter(x1, cutoffFreq, 4);
            y1 = butterworthFilter(y1, cutoffFreq, 4);
            a1 = butterworthFilter(a1, cutoffFreq, 4);
        } 
        else {
            // Adaptive frequency filtering
            x1 = adaptiveFrequencyFilter(x1);
            y1 = adaptiveFrequencyFilter(y1);
            a1 = adaptiveFrequencyFilter(a1);
        }
        
        // Update the input vectors with smoothed results
        x = x1;
        y = y1;
        a = a1;
    }

    // Professional adaptive radius calculation
    int Stabilizer::calculateAdaptiveRadius(const std::vector<float>& px, const std::vector<float>& py, const std::vector<float>& pa) {
        if(px.size() < 10) return params_.smoothingRadius;
        
        // Calculate motion variance over recent frames
        float varianceX = 0, varianceY = 0, varianceA = 0;
        float meanX = 0, meanY = 0, meanA = 0;
        
        size_t start = std::max(0, static_cast<int>(px.size()) - 20);
        size_t count = px.size() - start;
        
        // Calculate means
        for(size_t i = start; i < px.size(); i++) {
            meanX += px[i];
            meanY += py[i];
            meanA += pa[i];
        }
        meanX /= count;
        meanY /= count;
        meanA /= count;
        
        // Calculate variances
        for(size_t i = start; i < px.size(); i++) {
            varianceX += (px[i] - meanX) * (px[i] - meanX);
            varianceY += (py[i] - meanY) * (py[i] - meanY);
            varianceA += (pa[i] - meanA) * (pa[i] - meanA);
        }
        varianceX /= count;
        varianceY /= count;
        varianceA /= count;
        
        float totalVariance = std::sqrt(varianceX + varianceY + varianceA * 1000); // Scale rotation
        
        // Map variance to radius: high variance = high radius (more smoothing)
        int adaptiveRadius = static_cast<int>(std::max(5.0f, std::min(25.0f, totalVariance * 2.0f)));
        
        return adaptiveRadius;
    }

    // Advanced motion intent analysis (iPhone Action mode-like)
    MotionIntent Stabilizer::analyzeMotionIntent(const cv::Vec3f& motion, int frameIndex) {
        float magnitude = std::sqrt(motion[0] * motion[0] + motion[1] * motion[1]);
        float angularVel = std::abs(motion[2]) * 180.0f / M_PI * 30.0f; // Convert to deg/sec
        
        // Analyze motion pattern over recent frames
        if(transforms_.size() >= 15) {
            std::vector<float> recentMagnitudes;
            std::vector<float> recentDirections;
            
            for(int i = std::max(0, frameIndex - 15); i < frameIndex; i++) {
                if(i < static_cast<int>(transforms_.size())) {
                    cv::Vec3f t = transforms_[i];
                    float mag = std::sqrt(t[0] * t[0] + t[1] * t[1]);
                    float dir = std::atan2(t[1], t[0]);
                    
                    recentMagnitudes.push_back(mag);
                    recentDirections.push_back(dir);
                }
            }
            
            if(!recentMagnitudes.empty()) {
                // Check for consistent direction (panning)
                float directionVariance = calculateVariance(recentDirections);
                float magnitudeConsistency = calculateConsistency(recentMagnitudes);
                
                // Deliberate panning: consistent direction and magnitude
                if(directionVariance < 0.5f && magnitudeConsistency > 0.7f && magnitude > 5.0f) {
                    return MotionIntent::DELIBERATE_PAN;
                }
                
                // Shake: high frequency, low consistency
                if(magnitude < 3.0f && magnitudeConsistency < 0.3f && angularVel > 10.0f) {
                    return MotionIntent::SHAKE_REMOVAL;
                }
                
                // Action following: medium magnitude, variable direction
                if(magnitude > 3.0f && magnitude < 15.0f && directionVariance > 0.5f) {
                    return MotionIntent::FOLLOW_ACTION;
                }
            }
        }
        
        return MotionIntent::NORMAL;
    }

    // Calculate adaptive stabilization strength
    float Stabilizer::calculateAdaptiveStabilizationStrength(MotionIntent intent, const cv::Vec3f& motion) {
        float baseMagnitude = std::sqrt(motion[0] * motion[0] + motion[1] * motion[1]);
        
        // Base strength calculation
        float strength = 0.7f; // Default
        
        switch(intent) {
            case MotionIntent::DELIBERATE_PAN:
                strength = 0.1f + (baseMagnitude / 50.0f) * 0.2f; // Very light for panning
                break;
                
            case MotionIntent::SHAKE_REMOVAL:
                strength = 0.9f - (baseMagnitude / 10.0f) * 0.2f; // Strong for shake
                break;
                
            case MotionIntent::FOLLOW_ACTION:
                strength = 0.6f + (baseMagnitude / 20.0f) * 0.2f; // Balanced for action
                break;
                
            default:
                strength = 0.7f; // Standard stabilization
                break;
        }
        
        return std::max(0.1f, std::min(1.0f, strength));
    }

    // Utility function to calculate variance
    float Stabilizer::calculateVariance(const std::vector<float>& values) {
        if(values.empty()) return 0.0f;
        
        float mean = 0.0f;
        for(float v : values) mean += v;
        mean /= values.size();
        
        float variance = 0.0f;
        for(float v : values) {
            float diff = v - mean;
            variance += diff * diff;
        }
        variance /= values.size();
        
        return variance;
    }

    // Utility function to calculate consistency
    float Stabilizer::calculateConsistency(const std::vector<float>& values) {
        if(values.size() < 2) return 0.0f;
        
        float variance = calculateVariance(values);
        float mean = 0.0f;
        for(float v : values) mean += v;
        mean /= values.size();
        
        if(mean == 0.0f) return 0.0f;
        
        float consistency = 1.0f / (1.0f + (variance / (mean * mean)));
        return std::max(0.0f, std::min(1.0f, consistency));
    }

    // Professional motion validation and filtering (iPhone/GoPro-like)
    Transform Stabilizer::validateAndFilterMotion(const Transform& rawTransform) {
        frameCount_++;
        
        // Motion magnitude analysis
        float motionMagnitude = std::sqrt(rawTransform.dx * rawTransform.dx + 
                                        rawTransform.dy * rawTransform.dy);
        float angularMotion = std::abs(rawTransform.da) * 180.0f / M_PI;
        
        // Update motion history for pattern analysis
        MotionSample sample;
        sample.transform = rawTransform;
        sample.magnitude = motionMagnitude;
        sample.timestamp = frameCount_;
        sample.confidence = calculateMotionConfidence(rawTransform);
        
        motionHistory_.push_back(sample);
        if(motionHistory_.size() > 100) {
            motionHistory_.erase(motionHistory_.begin());
        }
        
        // Classify motion type (like iPhone Action mode)
        MotionType motionType = classifyMotion(rawTransform, motionMagnitude);
        
        // Scene-aware motion filtering
        Transform filteredTransform = rawTransform;
        
        // Apply different filters based on motion type
        switch(motionType) {
            case MotionType::INTENTIONAL_PAN:
                filteredTransform = applyIntentionalMotionFilter(rawTransform);
                break;
                
            case MotionType::CAMERA_SHAKE:
                filteredTransform = applyShakeRemovalFilter(rawTransform);
                break;
                
            case MotionType::WALKING_VIBRATION:
                filteredTransform = applyWalkingStabilization(rawTransform);
                break;
                
            case MotionType::VEHICLE_VIBRATION:
                filteredTransform = applyVehicleStabilization(rawTransform);
                break;
                
            default:
                filteredTransform = applyGeneralStabilization(rawTransform);
                break;
        }
        
        // Horizon lock if enabled
        if(horizonLockEnabled_) {
            filteredTransform = applyHorizonLock(filteredTransform);
        }
        
        // Update velocity and acceleration filters
        updatePredictiveFilters(filteredTransform);
        
        lastValidTransform_ = filteredTransform;
        return filteredTransform;
    }

    // Motion confidence calculation
    float Stabilizer::calculateMotionConfidence(const Transform& transform) {
        float magnitude = std::sqrt(transform.dx * transform.dx + transform.dy * transform.dy);
        float confidence = std::exp(-magnitude / 10.0f);  // Exponential decay
        return std::max(0.1f, std::min(1.0f, confidence));
    }

    // Professional motion classification (iPhone/GoPro-like)
    MotionType Stabilizer::classifyMotion(const Transform& transform, float magnitude) {
        float angularVelocity = std::abs(transform.da) * 180.0f / M_PI * 30.0f;  // Convert to deg/sec
        
        // Analyze motion history for patterns
        if(motionHistory_.size() >= 10) {
            bool consistentDirection = true;
            float avgDx = 0, avgDy = 0;
            
            for(size_t i = motionHistory_.size() - 10; i < motionHistory_.size(); i++) {
                avgDx += motionHistory_[i].transform.dx;
                avgDy += motionHistory_[i].transform.dy;
            }
            avgDx /= 10.0f;
            avgDy /= 10.0f;
            
            // Check for intentional panning motion
            float panSpeed = std::sqrt(avgDx * avgDx + avgDy * avgDy);
            if(panSpeed > 5.0f && angularVelocity < 10.0f) {
                return MotionType::INTENTIONAL_PAN;
            }
        }
        
        // Classify based on magnitude and angular velocity
        if(angularVelocity > 30.0f && magnitude < 5.0f) {
            return MotionType::CAMERA_SHAKE;
        }
        
        if(magnitude > 3.0f && magnitude < 8.0f && angularVelocity < 15.0f) {
            return MotionType::WALKING_VIBRATION;
        }
        
        if(magnitude > 10.0f) {
            return MotionType::VEHICLE_VIBRATION;
        }
        
        return MotionType::NORMAL;
    }

    // Intentional motion filter (preserve panning)
    Transform Stabilizer::applyIntentionalMotionFilter(const Transform& transform) {
        float smoothingFactor = 0.2f;  // Minimal smoothing
        
        Transform smoothed;
        smoothed.dx = transform.dx * (1.0f - smoothingFactor) + lastValidTransform_.dx * smoothingFactor;
        smoothed.dy = transform.dy * (1.0f - smoothingFactor) + lastValidTransform_.dy * smoothingFactor;
        smoothed.da = transform.da * (1.0f - smoothingFactor) + lastValidTransform_.da * smoothingFactor;
        
        return smoothed;
    }

    // Camera shake removal filter
    Transform Stabilizer::applyShakeRemovalFilter(const Transform& transform) {
        float smoothingFactor = 0.8f;  // Strong smoothing
        
        Transform smoothed;
        smoothed.dx = transform.dx * (1.0f - smoothingFactor) + lastValidTransform_.dx * smoothingFactor;
        smoothed.dy = transform.dy * (1.0f - smoothingFactor) + lastValidTransform_.dy * smoothingFactor;
        smoothed.da = transform.da * (1.0f - smoothingFactor) + lastValidTransform_.da * smoothingFactor;
        
        return smoothed;
    }

    // Walking stabilization filter
    Transform Stabilizer::applyWalkingStabilization(const Transform& transform) {
        float smoothingFactor = 0.5f;
        
        Transform smoothed;
        smoothed.dx = transform.dx * (1.0f - smoothingFactor) + lastValidTransform_.dx * smoothingFactor;
        smoothed.dy = transform.dy * (1.0f - smoothingFactor) + lastValidTransform_.dy * smoothingFactor;
        smoothed.da = transform.da * (1.0f - smoothingFactor) + lastValidTransform_.da * smoothingFactor;
        
        return smoothed;
    }

    // Vehicle stabilization filter
    Transform Stabilizer::applyVehicleStabilization(const Transform& transform) {
        float smoothingFactor = 0.7f;
        
        Transform smoothed;
        smoothed.dx = transform.dx * (1.0f - smoothingFactor) + lastValidTransform_.dx * smoothingFactor;
        smoothed.dy = transform.dy * (1.0f - smoothingFactor) + lastValidTransform_.dy * smoothingFactor;
        smoothed.da = transform.da * (1.0f - smoothingFactor) + lastValidTransform_.da * smoothingFactor;
        
        return smoothed;
    }

    // General stabilization filter
    Transform Stabilizer::applyGeneralStabilization(const Transform& transform) {
        float smoothingFactor = 0.6f;
        
        Transform smoothed;
        smoothed.dx = transform.dx * (1.0f - smoothingFactor) + lastValidTransform_.dx * smoothingFactor;
        smoothed.dy = transform.dy * (1.0f - smoothingFactor) + lastValidTransform_.dy * smoothingFactor;
        smoothed.da = transform.da * (1.0f - smoothingFactor) + lastValidTransform_.da * smoothingFactor;
        
        return smoothed;
    }

    // Horizon lock implementation
    Transform Stabilizer::applyHorizonLock(const Transform& transform) {
        // Estimate horizon angle from motion history
        if(motionHistory_.size() >= 30) {
            float totalRotation = 0;
            for(size_t i = motionHistory_.size() - 30; i < motionHistory_.size(); i++) {
                totalRotation += motionHistory_[i].transform.da;
            }
            
            // Update horizon estimate
            horizonAngle_ += totalRotation / 30.0f;
            horizonConfidence_ = std::min(1.0f, horizonConfidence_ + 0.01f);
            
            // Apply horizon correction
            Transform corrected = transform;
            if(horizonConfidence_ > 0.5f) {
                corrected.da = transform.da - horizonAngle_ * 0.1f;  // Gentle correction
            }
            
            return corrected;
        }
        
        return transform;
    }

    // Update predictive filters
    void Stabilizer::updatePredictiveFilters(const Transform& transform) {
        // Update velocity filter
        velocityFilter_.push_back(transform.dx);
        if(velocityFilter_.size() > 5) {
            velocityFilter_.erase(velocityFilter_.begin());
        }
        
        // Calculate acceleration
        if(velocityFilter_.size() >= 2) {
            float acceleration = velocityFilter_.back() - velocityFilter_[velocityFilter_.size()-2];
            accelerationFilter_.push_back(acceleration);
            if(accelerationFilter_.size() > 3) {
                accelerationFilter_.erase(accelerationFilter_.begin());
            }
        }
    }

    // Jetson Orin Nano specific optimizations
    void Stabilizer::optimizeForJetson() {
        // Apply Jetson-specific memory and processing optimizations
        if(params_.logging) logMessage("Applying Jetson Orin Nano optimizations", false);
        
        // Set optimal thread counts for Jetson
        cv::setNumThreads(4);  // Jetson Orin Nano has 6 cores, use 4 for OpenCV
        
        // Enable Jetson hardware acceleration hints
        cv::setUseOptimized(true);
        
        // Additional Jetson-specific optimizations
        #ifdef __ARM_NEON
        // ARM NEON optimizations are automatically enabled by OpenCV
        if(params_.logging) logMessage("ARM NEON optimizations enabled", false);
        #endif
        
        if(params_.logging) logMessage("Jetson optimizations applied", false);
    }

    // Butterworth filter implementation for frequency-specific smoothing
    std::vector<float> Stabilizer::butterworthFilter(const std::vector<float> &path, double cutoffFreq, int order) {
        if (path.empty() || order < 1) {
            return path;
        }
        
        std::vector<float> result(path.size());
        
        // Simple first-order butterworth implementation
        float alpha = static_cast<float>(cutoffFreq / (cutoffFreq + 1.0));
        
        // Forward pass (causal)
        result[0] = path[0];
        for (size_t i = 1; i < path.size(); i++) {
            result[i] = alpha * path[i] + (1.0f - alpha) * result[i-1];
        }
        
        // For higher order filters, repeat the process
        for (int o = 1; o < order; o++) {
            std::vector<float> temp = result;
            temp[0] = result[0];
            for (size_t i = 1; i < result.size(); i++) {
                temp[i] = alpha * result[i] + (1.0f - alpha) * temp[i-1];
            }
            result = temp;
        }
        
        return result;
    }
    
    // Adaptive frequency filter that removes different frequencies based on content
    std::vector<float> Stabilizer::adaptiveFrequencyFilter(const std::vector<float> &path) {
        if (path.empty()) {
            return path;
        }
        
        // HF: Use jitterFrequency parameter to determine filter cutoff
        float cutoffFreq = mapJitterFrequencyToCutoff(params_.jitterFrequency);
        
        if (params_.jitterFrequency == Parameters::ADAPTIVE) {
            // Original adaptive multi-stage filtering
            std::vector<float> highFreqFiltered = butterworthFilter(path, 0.3, 2);
            std::vector<float> medFreqFiltered = butterworthFilter(highFreqFiltered, 0.1, 2);
            std::vector<float> finalFiltered = butterworthFilter(medFreqFiltered, 0.05, 1);
            return finalFiltered;
        } else {
            // Single-stage filtering based on jitter frequency setting
            return butterworthFilter(path, cutoffFreq, 2);
        }
    }

    // Virtual Canvas Stabilization Implementation
    
    cv::Mat Stabilizer::applyVirtualCanvasStabilization(const cv::Mat& currentFrame, const cv::Vec3f& transform) {
        if (currentFrame.empty()) {
            return currentFrame;
        }
        
        // Initialize virtual canvas on first frame or size change
        if (virtualCanvas_.empty() || 
            virtualCanvas_.cols != static_cast<int>(currentFrame.cols * currentCanvasScale_) ||
            virtualCanvas_.rows != static_cast<int>(currentFrame.rows * currentCanvasScale_)) {
            
            // Calculate optimal canvas size based on recent motion
            if (params_.adaptiveCanvasSize && !transforms_.empty()) {
                currentCanvasScale_ = calculateOptimalCanvasSize(transform);
            } else {
                currentCanvasScale_ = params_.canvasScaleFactor;
            }
            
            canvasSize_ = cv::Size(
                static_cast<int>(currentFrame.cols * currentCanvasScale_),
                static_cast<int>(currentFrame.rows * currentCanvasScale_)
            );
            
            canvasCenter_ = cv::Point2f(canvasSize_.width / 2.0f, canvasSize_.height / 2.0f);
            
            // Create virtual canvas
            virtualCanvas_ = createVirtualCanvas(currentFrame, transform);
            
            // Create blending mask for seamless integration
            canvasBlendMask_ = cv::Mat::ones(canvasSize_, CV_32F);
            
            // Create smooth falloff at edges
            int edgeRadius = params_.edgeBlendRadius;
            for (int y = 0; y < canvasSize_.height; y++) {
                for (int x = 0; x < canvasSize_.width; x++) {
                    float distFromEdge = std::min({x, y, canvasSize_.width - x - 1, canvasSize_.height - y - 1});
                    if (distFromEdge < edgeRadius) {
                        float weight = distFromEdge / edgeRadius;
                        canvasBlendMask_.at<float>(y, x) = weight * weight; // Smooth quadratic falloff
                    }
                }
            }
        }
        
        // Update virtual canvas with current frame
        virtualCanvas_ = createVirtualCanvas(currentFrame, transform);
        
        // Blend temporal regions to eliminate jitter
        cv::Mat finalResult = blendTemporalRegions(virtualCanvas_, currentFrame, transform);
        
        // Extract the stabilized region from the center of the canvas
        cv::Point2f frameOffset(
            canvasCenter_.x - currentFrame.cols / 2.0f - transform[0],
            canvasCenter_.y - currentFrame.rows / 2.0f - transform[1]
        );
        
        cv::Rect extractRegion(
            std::max(0, static_cast<int>(frameOffset.x)),
            std::max(0, static_cast<int>(frameOffset.y)),
            currentFrame.cols,
            currentFrame.rows
        );
        
        // Ensure extract region is within canvas bounds
        extractRegion.x = std::min(extractRegion.x, finalResult.cols - extractRegion.width);
        extractRegion.y = std::min(extractRegion.y, finalResult.rows - extractRegion.height);
        extractRegion.width = std::min(extractRegion.width, finalResult.cols - extractRegion.x);
        extractRegion.height = std::min(extractRegion.height, finalResult.rows - extractRegion.y);
        
        if (extractRegion.width > 0 && extractRegion.height > 0 &&
            extractRegion.x >= 0 && extractRegion.y >= 0 &&
            extractRegion.x + extractRegion.width <= finalResult.cols &&
            extractRegion.y + extractRegion.height <= finalResult.rows) {
            
            cv::Mat stabilizedFrame = finalResult(extractRegion).clone();
            
            // Resize to original size if needed
            if (stabilizedFrame.size() != currentFrame.size()) {
                cv::resize(stabilizedFrame, stabilizedFrame, currentFrame.size(), 0, 0, cv::INTER_LANCZOS4);
            }
            
            return stabilizedFrame;
        }
        
        // Fallback to original frame if extraction fails
        return currentFrame;
    }
    
    void Stabilizer::updateTemporalFrameBuffer(const cv::Mat& frame, const cv::Vec3f& transform) {
        if (frame.empty()) return;
        
        // Add current frame to temporal buffer
        temporalFrameBuffer_.push_back(frame.clone());
        temporalTransformBuffer_.push_back(transform);
        
        // Maintain buffer size
        while (temporalFrameBuffer_.size() > static_cast<size_t>(params_.temporalBufferSize)) {
            temporalFrameBuffer_.pop_front();
            temporalTransformBuffer_.pop_front();
        }
        
        virtualCanvasFrameCount_++;
    }
    
    cv::Mat Stabilizer::createVirtualCanvas(const cv::Mat& currentFrame, const cv::Vec3f& transform) {
        if (currentFrame.empty()) {
            return cv::Mat();
        }
        
        // Create canvas filled with black initially
        cv::Mat canvas = cv::Mat::zeros(canvasSize_, currentFrame.type());
        
        // Place current frame at the center of canvas
        cv::Point2f framePos(
            canvasCenter_.x - currentFrame.cols / 2.0f,
            canvasCenter_.y - currentFrame.rows / 2.0f
        );
        
        cv::Rect frameRect(
            static_cast<int>(framePos.x),
            static_cast<int>(framePos.y),
            currentFrame.cols,
            currentFrame.rows
        );
        
        // Ensure frame rectangle is within canvas bounds
        cv::Rect canvasRect(0, 0, canvas.cols, canvas.rows);
        cv::Rect validRect = frameRect & canvasRect;
        
        if (validRect.width > 0 && validRect.height > 0) {
            // Calculate corresponding region in source frame
            cv::Rect srcRect(
                validRect.x - frameRect.x,
                validRect.y - frameRect.y,
                validRect.width,
                validRect.height
            );
            
            // Copy current frame to canvas
            if (srcRect.x >= 0 && srcRect.y >= 0 &&
                srcRect.x + srcRect.width <= currentFrame.cols &&
                srcRect.y + srcRect.height <= currentFrame.rows) {
                currentFrame(srcRect).copyTo(canvas(validRect));
            }
        }
        
        return canvas;
    }
    
    cv::Mat Stabilizer::blendTemporalRegions(const cv::Mat& canvas, const cv::Mat& currentFrame, const cv::Vec3f& transform) {
        cv::Mat result = canvas.clone();
        
        if (temporalFrameBuffer_.size() < 2) {
            return result; // Not enough temporal data
        }
        
        // Calculate regions that need filling
        std::vector<cv::Rect> emptyRegions;
        
        // Detect empty (black) regions in the canvas
        cv::Mat grayCanvas;
        cv::cvtColor(result, grayCanvas, cv::COLOR_BGR2GRAY);
        
        // Find contours of empty regions
        cv::Mat binary;
        cv::threshold(grayCanvas, binary, 1, 255, cv::THRESH_BINARY_INV);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Process significant empty regions
        for (const auto& contour : contours) {
            cv::Rect boundingRect = cv::boundingRect(contour);
            if (boundingRect.area() > 100) { // Only process significant regions
                emptyRegions.push_back(boundingRect);
            }
        }
        
        // Fill empty regions with content from temporal frames
        for (const cv::Rect& emptyRegion : emptyRegions) {
            cv::Mat bestFill;
            float bestWeight = 0.0f;
            
            // Search through temporal buffer for best matching content
            for (size_t i = 0; i < temporalFrameBuffer_.size() - 1; i++) { // Exclude current frame
                const cv::Mat& temporalFrame = temporalFrameBuffer_[i];
                const cv::Vec3f& frameTransform = temporalTransformBuffer_[i];
                
                // Calculate motion-compensated region
                cv::Vec3f relativeMotion = transform - frameTransform;
                
                if (isRegionAvailable(emptyRegion, relativeMotion, i)) {
                    cv::Mat temporalRegion = extractTemporalRegion(temporalFrame, emptyRegion, relativeMotion);
                    
                    if (!temporalRegion.empty()) {
                        // Calculate temporal weight (more recent = higher weight)
                        float temporalWeight = static_cast<float>(i + 1) / temporalFrameBuffer_.size();
                        temporalWeight *= params_.canvasBlendWeight;
                        
                        if (temporalWeight > bestWeight) {
                            bestFill = temporalRegion;
                            bestWeight = temporalWeight;
                        }
                    }
                }
            }
            
            // Apply best fill if found
            if (!bestFill.empty() && bestWeight > 0.0f) {
                seamlessBlend(result, bestFill, emptyRegion, bestWeight);
            }
        }
        
        return result;
    }
    
    float Stabilizer::calculateOptimalCanvasSize(const cv::Vec3f& recentMotion) {
        if (transforms_.empty()) {
            return params_.canvasScaleFactor;
        }
        
        // Analyze recent motion magnitude
        float motionMagnitude = std::sqrt(recentMotion[0] * recentMotion[0] + recentMotion[1] * recentMotion[1]);
        
        // Calculate motion variance over recent frames
        float maxMotion = 0.0f;
        int recentFrames = std::min(30, static_cast<int>(transforms_.size()));
        
        for (int i = transforms_.size() - recentFrames; i < static_cast<int>(transforms_.size()); i++) {
            if (i >= 0) {
                cv::Vec3f motion = transforms_[i];
                float magnitude = std::sqrt(motion[0] * motion[0] + motion[1] * motion[1]);
                maxMotion = std::max(maxMotion, magnitude);
            }
        }
        
        // Map motion to canvas scale
        float motionFactor = std::max(1.0f, maxMotion / 50.0f); // Normalize by expected max motion
        float optimalScale = params_.canvasScaleFactor + (motionFactor - 1.0f) * 0.5f;
        
        // Clamp to reasonable bounds
        optimalScale = std::max(params_.minCanvasScale, std::min(params_.maxCanvasScale, optimalScale));
        
        if (params_.logging) {
            logMessage("Adaptive canvas scale: " + std::to_string(optimalScale) + 
                      " (motion: " + std::to_string(maxMotion) + ")", false);
        }
        
        return optimalScale;
    }
    
    cv::Mat Stabilizer::extractTemporalRegion(const cv::Mat& frame, const cv::Rect& region, const cv::Vec3f& frameTransform) {
        if (frame.empty()) {
            return cv::Mat();
        }
        
        // Apply motion compensation to find corresponding region in temporal frame
        cv::Mat compensated = applyMotionCompensation(frame, frameTransform);
        
        // Calculate corresponding region with motion compensation
        cv::Rect compensatedRegion(
            region.x + static_cast<int>(frameTransform[0]),
            region.y + static_cast<int>(frameTransform[1]),
            region.width,
            region.height
        );
        
        // Ensure region is within frame bounds
        cv::Rect frameRect(0, 0, compensated.cols, compensated.rows);
        cv::Rect validRegion = compensatedRegion & frameRect;
        
        if (validRegion.width > 0 && validRegion.height > 0) {
            cv::Mat result = compensated(validRegion).clone();
            
            // Resize to match target region size if needed
            if (result.size() != region.size()) {
                cv::resize(result, result, region.size(), 0, 0, cv::INTER_LINEAR);
            }
            
            return result;
        }
        
        return cv::Mat();
    }
    
    void Stabilizer::seamlessBlend(cv::Mat& target, const cv::Mat& source, const cv::Rect& region, float weight) {
        if (source.empty() || target.empty()) return;
        
        // Ensure region is valid
        cv::Rect targetRect(0, 0, target.cols, target.rows);
        cv::Rect validRegion = region & targetRect;
        
        if (validRegion.width <= 0 || validRegion.height <= 0) return;
        
        // Resize source to match region if needed
        cv::Mat resizedSource = source;
        if (source.size() != validRegion.size()) {
            cv::resize(source, resizedSource, validRegion.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // Apply edge-aware blending
        cv::Mat targetRegion = target(validRegion);
        cv::Mat blendMask = cv::Mat::ones(validRegion.size(), CV_32F) * weight;
        
        // Create distance-based blending mask for smooth transitions
        int edgeRadius = std::min(params_.edgeBlendRadius, std::min(validRegion.width, validRegion.height) / 4);
        
        for (int y = 0; y < validRegion.height; y++) {
            for (int x = 0; x < validRegion.width; x++) {
                float distFromEdge = std::min({x, y, validRegion.width - x - 1, validRegion.height - y - 1});
                if (distFromEdge < edgeRadius) {
                    float edgeWeight = distFromEdge / edgeRadius;
                    blendMask.at<float>(y, x) *= edgeWeight;
                }
            }
        }
        
        // Perform alpha blending
        for (int y = 0; y < validRegion.height; y++) {
            for (int x = 0; x < validRegion.width; x++) {
                float alpha = blendMask.at<float>(y, x);
                cv::Vec3b targetPixel = targetRegion.at<cv::Vec3b>(y, x);
                cv::Vec3b sourcePixel = resizedSource.at<cv::Vec3b>(y, x);
                
                for (int c = 0; c < 3; c++) {
                    targetPixel[c] = static_cast<uchar>(
                        (1.0f - alpha) * targetPixel[c] + alpha * sourcePixel[c]
                    );
                }
                
                targetRegion.at<cv::Vec3b>(y, x) = targetPixel;
            }
        }
    }
    
    bool Stabilizer::isRegionAvailable(const cv::Rect& region, const cv::Vec3f& transform, int frameIndex) {
        // Check if the transformed region would be within the frame bounds
        cv::Rect transformedRegion(
            region.x + static_cast<int>(transform[0]),
            region.y + static_cast<int>(transform[1]),
            region.width,
            region.height
        );
        
        if (frameIndex >= static_cast<int>(temporalFrameBuffer_.size())) {
            return false;
        }
        
        const cv::Mat& frame = temporalFrameBuffer_[frameIndex];
        cv::Rect frameRect(0, 0, frame.cols, frame.rows);
        
        // Check if at least 50% of the region is available
        cv::Rect intersection = transformedRegion & frameRect;
        float coverage = static_cast<float>(intersection.area()) / static_cast<float>(region.area());
        
        return coverage > 0.5f;
    }
    
    cv::Mat Stabilizer::applyMotionCompensation(const cv::Mat& temporalFrame, const cv::Vec3f& motionVector) {
        if (temporalFrame.empty()) {
            return temporalFrame;
        }
        
        // Create inverse transformation matrix to compensate for motion
        float dx = -motionVector[0];
        float dy = -motionVector[1];
        float da = -motionVector[2];
        
        cv::Mat compensationMatrix = (cv::Mat_<float>(2, 3) <<
            std::cos(da), -std::sin(da), dx,
            std::sin(da), std::cos(da), dy
        );
        
        cv::Mat compensated;
        cv::warpAffine(temporalFrame, compensated, compensationMatrix, temporalFrame.size(),
                      cv::INTER_LINEAR, cv::BORDER_REFLECT);
        
        return compensated;
    }
    
    // HF: Drone high-frequency vibration suppression method implementations
    
    cv::Size Stabilizer::calculateDroneAnalysisSize(const cv::Mat& frame) {
        if (!params_.droneHighFreqMode) {
            return cv::Size(480, 270);  // Default analysis size
        }
        
        // Scale up analysis resolution while preserving aspect ratio
        int maxWidth = std::min(params_.hfAnalysisMaxWidth, frame.cols);
        float aspectRatio = static_cast<float>(frame.rows) / static_cast<float>(frame.cols);
        int height = static_cast<int>(maxWidth * aspectRatio);
        
        // Ensure even dimensions for better GPU performance
        maxWidth = (maxWidth / 2) * 2;
        height = (height / 2) * 2;
        
        if (params_.logging) {
            logMessage("HF: Using analysis resolution " + std::to_string(maxWidth) + "x" + std::to_string(height), false);
        }
        
        return cv::Size(maxWidth, height);
    }
    
    Transform Stabilizer::applyMicroShakeSuppression(const Transform& rawTransform) {
        Transform suppressed = rawTransform;
        
        // Calculate current median reference if we have enough history
        if (hfTranslationHistory_.size() >= 5) {
            hfMedianTranslation_ = calculateMedianTranslation();
        }
        
        // Calculate deviation from median reference
        cv::Vec2f currentTranslation(rawTransform.dx, rawTransform.dy);
        cv::Vec2f deviation = currentTranslation - hfMedianTranslation_;
        float magnitude = std::sqrt(deviation[0] * deviation[0] + deviation[1] * deviation[1]);
        
        // HF: More aggressive micro-shake suppression for freeze shot
        if (magnitude < params_.hfShakePx) {
            // For freeze shot, reduce residual to 1% instead of 10%
            cv::Vec2f residual = deviation * 0.01f;  // Much more aggressive
            suppressed.dx = hfMedianTranslation_[0] + residual[0];
            suppressed.dy = hfMedianTranslation_[1] + residual[1];
            
            if (params_.logging && magnitude > 0.1f) {
                logMessage("HF: Aggressive micro-shake suppressed, magnitude: " + std::to_string(magnitude), false);
            }
        } else if (magnitude < params_.hfShakePx * 2.0f) {
            // For slightly larger motions, still apply some suppression
            cv::Vec2f residual = deviation * 0.05f;  // 5% for medium micro-shakes
            suppressed.dx = hfMedianTranslation_[0] + residual[0];
            suppressed.dy = hfMedianTranslation_[1] + residual[1];
            
            if (params_.logging && magnitude > 0.5f) {
                logMessage("HF: Medium micro-shake suppressed, magnitude: " + std::to_string(magnitude), false);
            }
        }
        
        return suppressed;
    }
    
    Transform Stabilizer::applyRotationLowPass(const Transform& transform) {
        Transform filtered = transform;
        
        // Apply exponential low-pass filter to rotation instead of hard horizon lock
        if (params_.horizonLock) {
            hfRotationLowPass_ = (1.0f - params_.hfRotLPAlpha) * hfRotationLowPass_ + params_.hfRotLPAlpha * transform.da;
            filtered.da = hfRotationLowPass_;
            
            if (params_.logging && std::abs(transform.da - filtered.da) > 0.01f) {
                logMessage("HF: Rotation low-pass applied, original: " + std::to_string(transform.da) + 
                          ", filtered: " + std::to_string(filtered.da), false);
            }
        }
        
        return filtered;
    }
    
    void Stabilizer::updateTranslationHistory(const cv::Vec2f& translation) {
        hfTranslationHistory_.push_back(translation);
        
        // Keep only recent history (sliding window)
        if (hfTranslationHistory_.size() > 10) {
            hfTranslationHistory_.pop_front();
        }
    }
    
    cv::Vec2f Stabilizer::calculateMedianTranslation() {
        if (hfTranslationHistory_.empty()) {
            return cv::Vec2f(0.0f, 0.0f);
        }
        
        // Calculate median for x and y separately
        std::vector<float> xValues, yValues;
        for (const auto& trans : hfTranslationHistory_) {
            xValues.push_back(trans[0]);
            yValues.push_back(trans[1]);
        }
        
        std::sort(xValues.begin(), xValues.end());
        std::sort(yValues.begin(), yValues.end());
        
        size_t mid = xValues.size() / 2;
        float medianX = xValues.size() % 2 == 0 ? 
            (xValues[mid - 1] + xValues[mid]) / 2.0f : xValues[mid];
        float medianY = yValues.size() % 2 == 0 ? 
            (yValues[mid - 1] + yValues[mid]) / 2.0f : yValues[mid];
            
        return cv::Vec2f(medianX, medianY);
    }
    
    bool Stabilizer::shouldApplyConditionalCLAHE(int detectedFeatureCount) {
        if (!params_.enableConditionalCLAHE || !params_.droneHighFreqMode) {
            return false;
        }
        
        // Apply CLAHE when feature count is low (feature starvation)
        if (detectedFeatureCount >= 0 && detectedFeatureCount < 40) {
            hfFeatureStarvationCounter_++;
            return hfFeatureStarvationCounter_ > 2;  // Apply after consistent starvation
        } else {
            hfFeatureStarvationCounter_ = 0;
            return false;
        }
    }
    
    cv::Mat Stabilizer::applyConditionalCLAHE(const cv::Mat& grayFrame) {
        if (!shouldApplyConditionalCLAHE(-1) || grayFrame.empty()) {
            return grayFrame;
        }
        
        if (!hfConditionalCLAHE_) {
            hfConditionalCLAHE_ = cv::createCLAHE(2.0, cv::Size(8, 8));
        }
        
        cv::Mat enhanced;
        hfConditionalCLAHE_->apply(grayFrame, enhanced);
        
        if (params_.logging) {
            logMessage("HF: Conditional CLAHE applied for feature enhancement", false);
        }
        
        return enhanced;
    }
    
    float Stabilizer::mapJitterFrequencyToCutoff(Parameters::JitterFrequency freq) {
        switch (freq) {
            case Parameters::LOW:
                return 0.05f;      // Low frequency jitter
            case Parameters::MEDIUM:
                return 0.1f;       // Medium frequency jitter
            case Parameters::HIGH:
                return 0.25f;      // High frequency jitter (updated for drone props)
            case Parameters::ADAPTIVE:
                return 0.15f;      // Adaptive default
            default:
                return 0.1f;       // Fallback
        }
    }
    
    // HF: Dead zone freeze shot implementation
    Transform Stabilizer::applyDeadZoneFreeze(const Transform& rawTransform) {
        // Update motion accumulator
        updateMotionAccumulator(rawTransform);
        
        // Calculate current motion magnitude for decision making
        float currentMagnitude = std::sqrt(rawTransform.dx * rawTransform.dx + 
                                         rawTransform.dy * rawTransform.dy + 
                                         rawTransform.da * rawTransform.da * 100.0f);
        
        // Check if we should enter dead zone
        if (!hfInDeadZone_ && shouldEnterDeadZone(rawTransform)) {
            hfInDeadZone_ = true;
            hfFreezeCounter_ = params_.hfFreezeDuration;
            // Store the last valid transform instead of zero
            hfFrozenTransform_ = rawTransform;
            
            if (params_.logging) {
                logMessage("HF: Entering dead zone - camera freeze activated (motion: " + 
                          std::to_string(currentMagnitude) + ")", false);
            }
        }
        
        // If in dead zone, return frozen transform
        if (hfInDeadZone_) {
            hfFreezeCounter_--;
            
            // More lenient exit condition - either duration expired OR significant motion detected
            bool durationExpired = hfFreezeCounter_ <= 0;
            bool significantMotion = currentMagnitude > params_.hfDeadZoneThreshold * 1.5f; // Reduced from 2.0f
            bool accumulatedMotion = hfMotionAccumulator_ > params_.hfDeadZoneThreshold * 1.2f;
            
            if (durationExpired || significantMotion || accumulatedMotion) {
                hfInDeadZone_ = false;
                hfFreezeCounter_ = 0;
                hfMotionAccumulator_ = 0.0f; // Reset accumulator on exit
                
                if (params_.logging) {
                    logMessage("HF: Exiting dead zone - resuming motion tracking (motion: " + 
                              std::to_string(currentMagnitude) + ", accumulated: " + 
                              std::to_string(hfMotionAccumulator_) + ")", false);
                }
                
                return rawTransform;  // Resume normal motion
            }
            
            // Stay frozen - return minimal transform to maintain steady shot
            return Transform(0.0f, 0.0f, 0.0f);
        }
        
        return rawTransform;  // Normal processing
    }
    
    bool Stabilizer::shouldEnterDeadZone(const Transform& transform) {
        // Calculate motion magnitude
        float magnitude = std::sqrt(transform.dx * transform.dx + 
                                  transform.dy * transform.dy + 
                                  transform.da * transform.da * 100.0f);  // Scale rotation
        
        // Enter dead zone if motion is below threshold
        return magnitude < params_.hfDeadZoneThreshold;
    }
    
    void Stabilizer::updateMotionAccumulator(const Transform& transform) {
        // Calculate current motion magnitude
        float magnitude = std::sqrt(transform.dx * transform.dx + 
                                  transform.dy * transform.dy + 
                                  transform.da * transform.da * 100.0f);
        
        // Update accumulator with decay - but don't let it decay below current motion
        float decayedAccumulator = hfMotionAccumulator_ * params_.hfMotionAccumulatorDecay;
        hfMotionAccumulator_ = std::max(decayedAccumulator, magnitude);
        
        // Prevent accumulator from growing too large
        hfMotionAccumulator_ = std::min(hfMotionAccumulator_, params_.hfDeadZoneThreshold * 5.0f);
        
        // Clamp to reasonable range
        hfMotionAccumulator_ = std::max(0.0f, std::min(hfMotionAccumulator_, 100.0f));
    }
    
    Transform Stabilizer::getFrozenTransform() {
        return hfFrozenTransform_;
    }

} // namespace vs