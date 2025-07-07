// Filename: stabilizer.cpp (Single-file for demonstration)
// --------------------------------------------------------
// USAGE:
//   #include "stabilizer.cpp"  // or compile separately and link
//
// DESCRIPTION:
//   A fully self-contained C++ Stabilizer class that uses GPU/CPU pipelines.
//   The critical fix is the reshaping of detected corners to (1 x N, CV_32FC2)
//   so that cv::cuda::SparsePyrLKOpticalFlow::calc() doesn't segfault.
//
//   Dependencies:
//     - OpenCV with CUDA (for the GPU path): modules cudaarithm, cudaimgproc, cudaoptflow, cudafeatures2d, cudawarping
//     - Built with definitions like HAVE_OPENCV_CUDAARITHM, HAVE_OPENCV_CUDAOPTFLOW, etc.
//
//   Example usage in your main.cpp:
//     #include "stabilizer.cpp"
//
//     int main() {
//       Stabilizer::Parameters stabParams;
//       stabParams.useCuda = true;      // GPU mode
//       stabParams.logging = true;
//
//       Stabilizer stab(stabParams);
//
//       // For each frame read from your capture source:
//       //   cv::Mat stabilized = stab.stabilize(frameBGR);
//       //   if (!stabilized.empty()) { ... display or process ... }
//
//       return 0;
//     }

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <deque>
#include <string>
#include <vector>
#include <numeric>
#include <random>
#include "video/Stabilizer.h"

// If your OpenCV is built with these definitions, you can enable full GPU pipeline:
#if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
#include <opencv2/core/cuda.hpp>     // for cv::cuda::Stream
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#ifdef HAVE_OPENCV_CUDAOPTFLOW
#include <opencv2/cudaoptflow.hpp>
#endif
#ifdef HAVE_OPENCV_CUDAFEATURES2D
#include <opencv2/cudafeatures2d.hpp>
#endif
#include <opencv2/features2d.hpp>    // For CPU feature detectors

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
        if(params_.logging) logMessage("Initializing...");

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

        // Optimized smoothing radius for performance
        int effectiveRadius = std::max(3, std::min(params_.smoothingRadius, 15));
        
        // 1D box kernel - pre-allocated for efficiency
        boxKernel_.resize(effectiveRadius, 1.0f / effectiveRadius);
        
        // Pre-allocate vectors for better performance
        transforms_.reserve(100);
        path_.reserve(100);
        smoothedPath_.reserve(100);
        // Note: std::deque doesn't have reserve(), but it's still more efficient than std::vector for our use case

        // CPU CLAHE with optimized parameters
        claheCPU_ = cv::createCLAHE(1.5, cv::Size(8,8));  // Reduced clip limit for performance

    #if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
        if(useGpu_) {
            // GPU CLAHE
            claheGPU_ = cv::cuda::createCLAHE(2.0, cv::Size(8,8));
        }
        cudaStream_ = cv::cuda::Stream::Null();

    #endif

    #ifdef HAVE_OPENCV_CUDAOPTFLOW
        if(useGpu_) {
            // GPU SparsePyrLK
            pyrLK_ = cv::cuda::SparsePyrLKOpticalFlow::create();
            pyrLK_->setWinSize(cv::Size(21,21));
            pyrLK_->setMaxLevel(3);

    #ifdef HAVE_OPENCV_CUDAFEATURES2D
            // GPU GFTT
            gfttDetector_ = cv::cuda::createGoodFeaturesToTrackDetector(
                CV_8UC1, 
                params_.maxCorners, 
                params_.qualityLevel, 
                params_.minDistance,
                params_.blockSize,
                false,  // useHarris
                0.04
            );
    #endif
        }
    #endif
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
        if(claheGPU_) claheGPU_.release();
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
            // Initialize with first frame
            frameWidth_ = frame.cols;
            frameHeight_ = frame.rows;
            // Reduced analysis resolution for better performance on Jetson but not too aggressive
            constexpr int ANALYSIS_W = 640, ANALYSIS_H = 360;  // Restored for better stability
    double fx = static_cast<double>(ANALYSIS_W) / frame.cols;
    double fy = static_cast<double>(ANALYSIS_H) / frame.rows;
        cv::Mat firstSmall;
    cv::resize(frame, firstSmall,
               cv::Size(ANALYSIS_W, ANALYSIS_H),
               0, 0, cv::INTER_AREA);  // INTER_AREA for better quality

            // Convert to gray
            if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
                prevGrayGPU_ = convertColorAndEnhanceGPU(firstSmall);
    #endif
            } else {
                prevGrayCPU_ = convertColorAndEnhanceCPU(firstSmall);
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
        int effectiveRadius = std::max(3, std::min(params_.smoothingRadius, 15));
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
    // Balanced analysis resolution - good stability with reasonable performance
    constexpr int ANALYSIS_W = 640, ANALYSIS_H = 360;  // Restored for better stability
    cv::Mat small;
    cv::resize(currFrameBGR,              // full‑HD in
               small,                     // 640×360 out
               cv::Size(ANALYSIS_W, ANALYSIS_H),
               0, 0, cv::INTER_AREA);     // INTER_AREA for better quality
	analysisWidth_  = small.cols;   // 640
	analysisHeight_ = small.rows;   // 360

        // Convert current frame to Gray
        cv::Mat currGrayCPU;
    #ifdef HAVE_OPENCV_CUDAARITHM
        cv::cuda::GpuMat currGrayGPU;
    #endif

        if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
            currGrayGPU = convertColorAndEnhanceGPU(small);
    #endif
        } else {
            currGrayCPU = convertColorAndEnhanceCPU(small);
        }

        // Optical flow
        if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAOPTFLOW
            if(prevPtsGPU_.empty()) {
                // no prev points => transform=0
                transforms_.push_back(cv::Vec3f(0.f,0.f,0.f));
            }
            else {
                cv::cuda::GpuMat currPtsGPU, statusGPU, errGPU;
                // calc
                pyrLK_->calc(
                    prevGrayGPU_,
                    currGrayGPU,
                    prevPtsGPU_,
                    currPtsGPU,
                    statusGPU,
                    errGPU
                );
                // The output currPtsGPU is 1xN, CV_32FC2
                // download to CPU
                std::vector<cv::Point2f> currPointsCPU;
                if(!currPtsGPU.empty()) {
                    currPointsCPU.resize(currPtsGPU.cols);
                    currPtsGPU.download(currPointsCPU);
                }
                std::vector<uchar> statusCPU;
                if(!statusGPU.empty()) {
                    statusCPU.resize(statusGPU.cols);
                    statusGPU.download(statusCPU);
                }

                // filter
                std::vector<cv::Point2f> validPrev, validCurr;
                for(size_t i=0; i<statusCPU.size(); i++){
                    if(statusCPU[i]) {
                        validPrev.push_back(prevKeypointsCPU_[i]);
                        validCurr.push_back(currPointsCPU[i]);
                    }
                }
                
                // Apply SightLine-inspired outlier rejection - simplified for performance
                if (params_.outlierRejection && validPrev.size() > 15) {  // Increased threshold
                    filterOutliers(validPrev, validCurr);
                }
                
                // estimate affine transform
                cv::Mat T = cv::Mat::eye(2,3,CV_64F);
                if(validPrev.size() >= 6 && validCurr.size() >= 6) {  // Need minimum 6 points
                    cv::Mat affine = cv::estimateAffinePartial2D(validPrev, validCurr, cv::noArray(), cv::RANSAC, 3.0);
                    if(!affine.empty()) {
                        T = affine;
                    }
                }
                double dx = T.at<double>(0,2);
                double dy = T.at<double>(1,2);
                double da = std::atan2(T.at<double>(1,0), T.at<double>(0,0));
                
                // Skip horizon lock for performance unless specifically enabled
                if (params_.horizonLock) {
                    da = 0.0;
                }
                
                transforms_.push_back(cv::Vec3f((float)dx,(float)dy,(float)da));
            }
    #endif
        }
        else {
            // CPU Optical flow
            if(!prevKeypointsCPU_.empty()) {
                std::vector<cv::Point2f> tmpCurr;
                std::vector<uchar> status;
                std::vector<float> err;
                cv::calcOpticalFlowPyrLK(
                    prevGrayCPU_, currGrayCPU,
                    prevKeypointsCPU_,
                    tmpCurr,
                    status, err
                );
                std::vector<cv::Point2f> validPrev, validCurr;
                for(size_t i=0; i<status.size(); i++) {
                    if(status[i]) {
                        validPrev.push_back(prevKeypointsCPU_[i]);
                        validCurr.push_back(tmpCurr[i]);
                    }
                }
                
                // Apply SightLine-inspired outlier rejection - simplified for performance
                if (params_.outlierRejection && validPrev.size() > 15) {  // Increased threshold
                    filterOutliers(validPrev, validCurr);
                }
                
                cv::Mat T = cv::Mat::eye(2,3,CV_64F);
                if(validPrev.size() >= 6 && validCurr.size() >= 6) {  // Need minimum 6 points
                    cv::Mat affine = cv::estimateAffinePartial2D(validPrev, validCurr, cv::noArray(), cv::RANSAC, 3.0);
                    if(!affine.empty()) {
                        T = affine;
                    }
                }
                double dx = T.at<double>(0,2);
                double dy = T.at<double>(1,2);
                double da = std::atan2(T.at<double>(1,0), T.at<double>(0,0));
                
                // Skip horizon lock for performance unless specifically enabled
                if (params_.horizonLock) {
                    da = 0.0;
                }
                
                transforms_.push_back(cv::Vec3f((float)dx,(float)dy,(float)da));
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
        // if (params_.adaptiveSmoothing) {
        //     updateAdaptiveParameters();
        // }

        // Now detect features on current for next iteration
        if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
            cv::cuda::GpuMat currGray = currGrayGPU;
    #ifdef HAVE_OPENCV_CUDAFEATURES2D
            if(gfttDetector_) {
                // Use enhanced GPU feature detection
                cv::cuda::GpuMat cornersGPU = detectFeaturesGPU(currGray);
                
                // If valid corners detected
                if(!cornersGPU.empty()) {
                    std::vector<cv::Point2f> cornersCPU(cornersGPU.cols);
                    cornersGPU.download(cornersCPU);

                    prevKeypointsCPU_ = cornersCPU;
    #ifdef HAVE_OPENCV_CUDAOPTFLOW
                    if(!cornersCPU.empty()) {
                        cv::Mat ptsMat(1, (int)cornersCPU.size(), CV_32FC2, (void*)cornersCPU.data());
                        prevPtsGPU_.upload(ptsMat);
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
            // Use enhanced CPU feature detection
            prevKeypointsCPU_ = detectFeatures(currGrayCPU);
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

        if((size_t)oldestIdx >= transforms_.size()) {
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
        
        // Apply the selected smoothing method - optimized for performance
        std::vector<float> sx, sy, sa;
        if (params_.smoothingMethod == "gaussian") {
            // Apply Gaussian smoothing only if explicitly requested
            sx = gaussianFilterConvolve(px, params_.gaussianSigma);
            sy = gaussianFilterConvolve(py, params_.gaussianSigma);
            sa = gaussianFilterConvolve(pa, params_.gaussianSigma);
        }
        else {
            // Default to box filter smoothing - most efficient for real-time
            sx = boxFilterConvolve(px);
            sy = boxFilterConvolve(py);
            sa = boxFilterConvolve(pa);
        }

        smoothedPath_.resize(path_.size());
        for(size_t i=0; i<path_.size(); i++) {
            smoothedPath_[i] = cv::Vec3f(sx[i], sy[i], sa[i]);
        }

        // Apply motion prediction for better stability
        cv::Vec3f raw = transforms_[oldestIdx];
        cv::Vec3f diff = smoothedPath_[oldestIdx] - path_[oldestIdx];
        
        if (params_.motionPrediction && oldestIdx > 0) {
            // Check if the current motion is intentional or not
            bool intentional = isIntentionalMotion(raw);
            
            if (intentional) {
                // For intentional motion, reduce smoothing effect to be more responsive
                diff *= 0.7f;  // Apply 70% of the correction for intentional motion
                if (params_.logging) {
                    logMessage("Motion determined to be intentional - reducing stabilization effect");
                }
            } else {
                // For unintentional motion (camera shake), apply full stabilization
                diff *= 1.0f;  // Apply full correction
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

        // Warp the frame using the transformation matrix
        cv::Mat stabilized;
        if(useGpu_) {
    #if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
            cv::cuda::GpuMat gpuIn, gpuOut;
            gpuIn.upload(frameWithBorder, cudaStream_);
            
            // For fade border, use BORDER_CONSTANT
            int actualBorderMode = (params_.borderType == "fade") ? cv::BORDER_CONSTANT : borderMode_;
            
            cv::cuda::warpAffine(
                gpuIn, gpuOut, T, gpuIn.size(),
                cv::INTER_LINEAR, actualBorderMode, cv::Scalar(), cudaStream_
            );
            gpuOut.download(stabilized, cudaStream_);
            cudaStream_.waitForCompletion();
    #else
            // fallback CPU
            int actualBorderMode = (params_.borderType == "fade") ? cv::BORDER_CONSTANT : borderMode_;
            
            cv::warpAffine(
                frameWithBorder, stabilized,
                T, frameWithBorder.size(),
                cv::INTER_LINEAR, actualBorderMode
            );
    #endif
        } else {
            int actualBorderMode = (params_.borderType == "fade") ? cv::BORDER_CONSTANT : borderMode_;
            
            cv::warpAffine(
                frameWithBorder, stabilized,
                T, frameWithBorder.size(),
                cv::INTER_LINEAR, actualBorderMode
            );
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

        return stabilized;
    }

std::vector<float> Stabilizer::boxFilterConvolve(const std::vector<float> &path)
{
    if(path.empty()) return {};
    
    // Use proper smoothing radius for stability
    int r = std::max(5, std::min(params_.smoothingRadius, 50));  // Allow larger radius for stability

    if(path.size() <= r) {
        // For very small arrays, use mean
        float sum = std::accumulate(path.begin(), path.end(), 0.0f);
        float mean = sum / path.size();
        std::vector<float> result(path.size(), mean);
        return result;
    }
    
    // Use proper padding for better edge handling
    std::vector<float> padded(path.size() + 2*r);
    
    // Pad with edge values (replicate border)
    for(int i = 0; i < r; i++) {
        padded[i] = path[0];  // Left padding
        padded[padded.size() - 1 - i] = path[path.size() - 1];  // Right padding
    }
    
    // Copy original data
    for(size_t i = 0; i < path.size(); i++) {
        padded[i + r] = path[i];
    }
    
    // Apply box filter with proper normalization
    std::vector<float> result(path.size());
    int kernelSize = 2 * r + 1;
    
    for(size_t i = 0; i < path.size(); i++) {
        float sum = 0.0f;
        for(int j = 0; j < kernelSize; j++) {
            sum += padded[i + j];
        }
        result[i] = sum / kernelSize;
    }
    
    return result;
}

    // CPU color + CLAHE
    cv::Mat Stabilizer::convertColorAndEnhanceCPU(const cv::Mat &frameBGR)
    {
        cv::Mat gray;
        cv::cvtColor(frameBGR, gray, cv::COLOR_BGR2GRAY);
        cv::Mat eq;
        claheCPU_->apply(gray, eq);
        return eq;
    }

    #ifdef HAVE_OPENCV_CUDAARITHM
    // GPU color + CLAHE
    cv::cuda::GpuMat Stabilizer::convertColorAndEnhanceGPU(const cv::Mat &frameBGR)
    {
        cv::cuda::GpuMat gpuIn, gpuGray, gpuCLAHE;
        gpuIn.upload(frameBGR, cudaStream_);
        cv::cuda::cvtColor(gpuIn, gpuGray, cv::COLOR_BGR2GRAY, 0, cudaStream_);
        if(claheGPU_) {
            claheGPU_->apply(gpuGray, gpuCLAHE);
        } else {
            gpuCLAHE = gpuGray;
        }
        cudaStream_.waitForCompletion();
        return gpuCLAHE;
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
            // Apply specific frequency filter based on jitter type
            double cutoffFreq = 0.1; // Default medium
            
            switch (params_.jitterFrequency) {
                case Parameters::LOW:
                    cutoffFreq = 0.05; // Lower cutoff for slow oscillations
                    break;
                case Parameters::HIGH:
                    cutoffFreq = 0.3; // Higher cutoff for vibrations
                    break;
                default:
                    break; // Use default for MEDIUM
            }
            
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
        
        // Stage 3: Second pass smoothing with stageTwoRadius for extra stability
        if (params_.smoothingMethod == "gaussian") {
            // Use stronger smoothing for the second pass
            double sigma2 = params_.gaussianSigma * 1.5;
            x = gaussianFilterConvolve(x1, sigma2);
            y = gaussianFilterConvolve(y1, sigma2);
            a = gaussianFilterConvolve(a1, sigma2);
        } 
        else if (params_.smoothingMethod == "kalman") {
            // For Kalman, just apply a second pass
            x = kalmanFilterSmooth(x1);
            y = kalmanFilterSmooth(y1);
            a = kalmanFilterSmooth(a1);
        }
        else {
            // Use stage two radius for second pass with box filter
            int originalRadius = params_.smoothingRadius;
            params_.smoothingRadius = params_.stageTwoRadius;
            boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);
            
            x = boxFilterConvolve(x1);
            y = boxFilterConvolve(y1);
            a = boxFilterConvolve(a1);
            
            // Restore
            params_.smoothingRadius = originalRadius;
            boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);
        }
        
        // Stage 4: Apply temporal filtering if enabled
        if (params_.useTemporalFiltering) {
            x = temporalFilter(x);
            y = temporalFilter(y);
            a = temporalFilter(a);
        }
    }
    
    // Temporal filtering across multiple frames for even smoother results
    std::vector<float> Stabilizer::temporalFilter(const std::vector<float> &path)
    {
        static std::deque<std::vector<float>> pathHistory;
        
        // Store the current path in history
        pathHistory.push_back(path);
        
        // Keep history limited to window size
        while (pathHistory.size() > static_cast<size_t>(params_.temporalWindowSize)) {
            pathHistory.pop_front();
        }
        
        // If we don't have enough history, return original path
        if (pathHistory.size() < 3) {
            return path;
        }
        
        // Apply temporal median filtering
        std::vector<float> result(path.size());
        
        for (size_t i = 0; i < path.size(); i++) {
            std::vector<float> values;
            values.reserve(pathHistory.size());
            
            for (const auto &historicalPath : pathHistory) {
                if (i < historicalPath.size()) {
                    values.push_back(historicalPath[i]);
                }
            }
            
            if (!values.empty()) {
                // Sort and get median value
                std::sort(values.begin(), values.end());
                size_t mid = values.size() / 2;
                
                if (values.size() % 2 == 0) {
                    // Average of two middle values
                    result[i] = (values[mid-1] + values[mid]) / 2.0f;
                } else {
                    // Exact middle value
                    result[i] = values[mid];
                }
            } else {
                result[i] = path[i];
            }
        }
        
        return result;
    }
    
    // Dynamically determine optimal border size based on motion magnitude
    float Stabilizer::calculateDynamicBorderSize(const std::vector<cv::Vec3f> &recentTransforms)
    {
        if (recentTransforms.empty()) {
            return static_cast<float>(params_.borderSize);
        }
        
        // Calculate maximum motion over recent frames
        float maxMotion = 0.0f;
        for (const auto &transform : recentTransforms) {
            float motionMagnitude = std::sqrt(transform[0] * transform[0] + transform[1] * transform[1]);
            maxMotion = std::max(maxMotion, motionMagnitude);
        }
        
        // Scale border based on detected motion
        float dynamicBorder = params_.borderSize;
        
        if (maxMotion > params_.motionThresholdLow) {
            // Scale border proportionally to motion
            float motionFactor = std::min(static_cast<float>(maxMotion / params_.motionThresholdHigh), 1.0f);
            dynamicBorder = params_.borderSize * (1.0f + motionFactor * (params_.borderScaleFactor - 1.0f));
        }
        
        return dynamicBorder;
    }
    
    // Roll compensation for banking during turns (common in flight footage)
    void Stabilizer::compensateForRoll(cv::Vec3f &tform)
    {
        // Extract rotation component
        float rotation = tform[2];
        
        // Check for significant roll (banking)
        if (std::abs(rotation) > 0.05f) { // ~3 degrees threshold
            // Apply compensation factor to allow some natural banking
            // while still reducing excessive roll
            tform[2] = rotation * (1.0f - params_.rollCompensationFactor);
            
            if (params_.logging) {
                logMessage("Compensating for roll/banking: " + std::to_string(rotation) + 
                           " reduced to " + std::to_string(tform[2]));
            }
        }
    }
    
    // Butterworth filter implementation for frequency-specific smoothing
    std::vector<float> Stabilizer::butterworthFilter(const std::vector<float> &path, double cutoffFreq, int order)
    {
        if (path.empty() || order < 1) {
            return path;
        }
        
        std::vector<float> result(path.size());
        
        // Simple first-order butterworth implementation
        // A more complete implementation would use proper digital filter design
        float alpha = cutoffFreq / (cutoffFreq + 1.0f);
        
        // Forward pass (causal)
        result[0] = path[0];
        for (size_t i = 1; i < path.size(); i++) {
            result[i] = alpha * path[i] + (1.0f - alpha) * result[i-1];
        }
        
        // For higher order filters, repeat the process
        for (int o = 1; o < order; o++) {
            std::vector<float> temp = result;
            
            // Forward pass
            for (size_t i = 1; i < temp.size(); i++) {
                result[i] = alpha * temp[i] + (1.0f - alpha) * result[i-1];
            }
        }
        
        return result;
    }
    
    // Adaptive frequency filter that removes different frequencies based on content
    std::vector<float> Stabilizer::adaptiveFrequencyFilter(const std::vector<float> &path)
    {
        if (path.empty()) {
            return path;
        }
        
        // First apply a high-frequency filter to remove vibrations
        std::vector<float> highFreqFiltered = butterworthFilter(path, 0.3, 2);
        
        // Then apply a medium-frequency filter to smooth general motion
        std::vector<float> medFreqFiltered = butterworthFilter(highFreqFiltered, 0.1, 2);
        
        // Finally apply a very gentle low-frequency filter to preserve intentional motion
        std::vector<float> finalFiltered = butterworthFilter(medFreqFiltered, 0.05, 1);
        
        return finalFiltered;
    }
    
    // Separate motion into translation and rotation components for independent processing
    void Stabilizer::separateMotionComponents(const cv::Vec3f &motion, cv::Vec2f &translation, float &rotation)
    {
        // Extract translation components
        translation[0] = motion[0]; // x translation
        translation[1] = motion[1]; // y translation
        
        // Extract rotation component
        rotation = motion[2]; // rotation angle
        
        if (params_.logging) {
            logMessage("Motion separated: translation(" + 
                       std::to_string(translation[0]) + ", " + 
                       std::to_string(translation[1]) + "), rotation=" + 
                       std::to_string(rotation));
        }
    }
    
    // Calculate reliability score for features based on tracking consistency
    float Stabilizer::getFeatureReliabilityScore(const std::vector<cv::Point2f> &prevPts, const std::vector<cv::Point2f> &currPts)
    {
        if (prevPts.empty() || currPts.empty() || prevPts.size() != currPts.size()) {
            return 0.0f;
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
        
        // Calculate variance (as a measure of consistency)
        float variance = 0.0f;
        for (const auto& m : motions) {
            float dx = m.x - meanMotion.x;
            float dy = m.y - meanMotion.y;
            variance += dx*dx + dy*dy;
        }
        variance /= motions.size();
        
        // Convert variance to reliability score (inverse relationship)
        // Lower variance means higher reliability
        float reliability = 1.0f / (1.0f + variance);
        
        return reliability;
    }
    
    // Classify scene type to adapt parameters
    void Stabilizer::classifySceneType(const cv::Mat &frame)
    {
        // Simple classification based on recent motion patterns and image characteristics
        
        // Skip if not enough history
        if (transforms_.size() < 5) {
            return;
        }
        
        // Calculate average motion magnitude over recent frames
        float avgMotion = 0.0f;
        float avgRotation = 0.0f;
        int recentFrames = std::min(5, static_cast<int>(transforms_.size()));
        
        for (int i = 0; i < recentFrames; i++) {
            int idx = transforms_.size() - 1 - i;
            cv::Vec3f transform = transforms_[idx];
            float magnitude = std::sqrt(transform[0] * transform[0] + transform[1] * transform[1]);
            avgMotion += magnitude;
            avgRotation += std::abs(transform[2]);
        }
        
        avgMotion /= recentFrames;
        avgRotation /= recentFrames;
        
        // Analyze image content (simple edge detection as proxy for scene complexity)
        cv::Mat gray, edges;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame.clone();
        }
        
        cv::Canny(gray, edges, 100, 200);
        double edgeRatio = cv::countNonZero(edges) / static_cast<double>(edges.total());
        
        // Scene classification based on motion and content
        if (avgMotion < 2.0f && edgeRatio < 0.05) {
            // Static scene with few features
            if (params_.logging) {
                logMessage("Scene classified as: Static, low-detail");
            }
        } 
        else if (avgMotion < 2.0f && edgeRatio >= 0.05) {
            // Static scene with many features
            if (params_.logging) {
                logMessage("Scene classified as: Static, high-detail");
            }
        }
        else if (avgMotion >= 2.0f && avgMotion < 10.0f && avgRotation < 0.02) {
            // Slow panning scene
            if (params_.logging) {
                logMessage("Scene classified as: Slow panning");
            }
        }
        else if (avgMotion >= 10.0f || avgRotation >= 0.02) {
            // Fast motion scene
            if (params_.logging) {
                logMessage("Scene classified as: Fast motion");
            }
        }
    }
    
    // Adjust parameters based on detected scene type
    void Stabilizer::adjustParametersForScene()
    {
        // This would be called after classifySceneType()
        // Assume scene classification data is stored in member variables
        
        // For now, use a simple approach based just on recent motion
        if (transforms_.size() < 5) {
            return;
        }
        
        // Calculate average motion magnitude over recent frames
        float avgMotion = 0.0f;
        float avgRotation = 0.0f;
        int recentFrames = std::min(5, static_cast<int>(transforms_.size()));
        
        for (int i = 0; i < recentFrames; i++) {
            int idx = transforms_.size() - 1 - i;
            cv::Vec3f transform = transforms_[idx];
            float magnitude = std::sqrt(transform[0] * transform[0] + transform[1] * transform[1]);
            avgMotion += magnitude;
            avgRotation += std::abs(transform[2]);
        }
        
        avgMotion /= recentFrames;
        avgRotation /= recentFrames;
        
        // Adjust parameters based on motion patterns
        if (avgMotion < 2.0f) {
            // Low motion - use stronger smoothing
            if (params_.adaptiveSmoothing) {
                int newRadius = params_.maxSmoothingRadius;
                if (newRadius != params_.smoothingRadius) {
                    params_.smoothingRadius = newRadius;
                    boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);
                    if (params_.logging) {
                        logMessage("Scene-based parameter adjustment: Low motion, increasing smoothing radius to " + 
                                   std::to_string(params_.smoothingRadius));
                    }
                }
            }
        }
        else if (avgMotion >= 10.0f || avgRotation >= 0.02) {
            // High motion - use less smoothing for responsiveness
            if (params_.adaptiveSmoothing) {
                int newRadius = params_.minSmoothingRadius;
                if (newRadius != params_.smoothingRadius) {
                    params_.smoothingRadius = newRadius;
                    boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);
                    if (params_.logging) {
                        logMessage("Scene-based parameter adjustment: High motion, decreasing smoothing radius to " + 
                                   std::to_string(params_.smoothingRadius));
                    }
                }
            }
        }
    }
    
    // Apply deep learning based stabilization 
    cv::Mat Stabilizer::applyDeepStabilization(const cv::Mat &frame)
    {
        // This function would integrate with a deep learning model for stabilization
        // For now, provide a placeholder implementation
        
        if (!params_.deepStabilization || frame.empty() || params_.modelPath.empty()) {
            return frame.clone();
        }
        
        if (params_.logging) {
            logMessage("Deep stabilization requested but not fully implemented");
        }
        
        // In a real implementation, this would:
        // 1. Load the DNN model from params_.modelPath
        // 2. Prepare the input frame
        // 3. Run inference
        // 4. Process the output
        
        // For now, just return the original frame
        return frame.clone();
    }
}  // namespace vs
