// Robust Digital Stabilizer - Focused on Shake Avoidance
// Clean implementation optimized for Jetson Orin Nano
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

    // Core shake detection and suppression constants
    static constexpr float SHAKE_THRESHOLD_PX = 3.0f;      // Translation shake threshold in pixels
    static constexpr float ROTATION_SHAKE_RAD = 0.03f;     // Rotation shake threshold in radians (~1.7 degrees)
    static constexpr float SHAKE_DAMPING_FACTOR = 0.15f;   // How much to dampen detected shake
    static constexpr int MIN_TRACKING_FEATURES = 30;       // Minimum features for reliable tracking
    static constexpr float OUTLIER_THRESHOLD = 15.0f;      // Outlier rejection threshold in pixels

    // Helper function for logging
    void Stabilizer::logMessage(const std::string& msg, bool isError) const {
        if (params_.logging) {
            if (isError) {
                std::cerr << "[Stabilizer ERROR] " << msg << std::endl;
            } else {
                std::cout << "[Stabilizer] " << msg << std::endl;
            }
        }
    }

    // Constructor - simplified and focused
    Stabilizer::Stabilizer(const Parameters &params) : params_(params) {
        logMessage("Initializing robust shake-avoiding stabilizer");
        
        useGpu_ = params_.useCuda;
        firstFrame_ = true;
        nextFrameIndex_ = 0;
        frameWidth_ = 0;
        frameHeight_ = 0;
        
        // Essential data structures
        transforms_.reserve(500);
        path_.reserve(500);
        smoothedPath_.reserve(500);
        
        // Simple box filter for trajectory smoothing
        int radius = std::max(5, std::min(params_.smoothingRadius, 30));
        boxKernel_.resize(radius, 1.0f / radius);
        
        // Initialize GPU optical flow if available
        #ifdef HAVE_OPENCV_CUDAOPTFLOW
        if (useGpu_) {
            try {
                auto pyrLK = cv::cuda::SparsePyrLKOpticalFlow::create();
                pyrLK->setWinSize(cv::Size(21, 21));
                pyrLK->setMaxLevel(3);
                logMessage("GPU optical flow initialized");
            } catch (const cv::Exception& e) {
                logMessage("GPU optical flow failed, falling back to CPU", true);
                useGpu_ = false;
            }
        }
        #endif
        
        logMessage("Stabilizer initialized with radius: " + std::to_string(radius));
    }

    Stabilizer::~Stabilizer() {
        clean();
    }

    void Stabilizer::clean() {
        frameQueue_.clear();
        frameIndexQueue_.clear();
        transforms_.clear();
        path_.clear();
        smoothedPath_.clear();
        prevGrayCPU_.release();
        prevKeypointsCPU_.clear();
        
        #ifdef HAVE_OPENCV_CUDAARITHM
        // Clean GPU resources if used
        #endif
        
        firstFrame_ = true;
        nextFrameIndex_ = 0;
        frameWidth_ = 0;
        frameHeight_ = 0;
        origSize_ = cv::Size();
        
        logMessage("Stabilizer cleaned");
    }

    cv::Mat Stabilizer::stabilize(const cv::Mat &frame) {
        if (frame.empty()) {
            logMessage("Empty frame received", true);
            return cv::Mat();
        }

        if (firstFrame_) {
            return initializeFirstFrame(frame);
        }

        // Store frame for processing
        frameQueue_.push_back(frame.clone());
        frameIndexQueue_.push_back(nextFrameIndex_);
        
        // Calculate transformation to current frame
        generateTransform(frame);

        // Apply smoothing if we have enough frames
        int radius = std::min(params_.smoothingRadius, 30);
        if (frameIndexQueue_.size() < static_cast<size_t>(radius)) {
            nextFrameIndex_++;
            return cv::Mat(); // Not ready yet
        }

        cv::Mat stabilized = applyNextSmoothTransform();
        nextFrameIndex_++;
        return stabilized;
    }

    cv::Mat Stabilizer::flush() {
        if (frameQueue_.empty()) {
            return cv::Mat();
        }
        return applyNextSmoothTransform();
    }

    cv::Mat Stabilizer::initializeFirstFrame(const cv::Mat &frame) {
        frameWidth_ = frame.cols;
        frameHeight_ = frame.rows;
        origSize_ = frame.size();
        
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        prevGrayCPU_ = gray.clone();
        
        // Detect initial features
        detectInitialFeatures(gray);
        
        firstFrame_ = false;
        logMessage("Initialized with " + std::to_string(prevKeypointsCPU_.size()) + " features");
        
        return frame.clone(); // Return first frame as-is
    }

    void Stabilizer::detectInitialFeatures(const cv::Mat &gray) {
        prevKeypointsCPU_.clear();
        
        // Use goodFeaturesToTrack for reliable corner detection
        cv::goodFeaturesToTrack(
            gray, prevKeypointsCPU_,
            params_.maxCorners,
            params_.qualityLevel,
            params_.minDistance,
            cv::noArray(),
            params_.blockSize,
            false, // useHarrisDetector
            0.04   // k
        );
        
        // Filter features away from borders for stability
        std::vector<cv::Point2f> filteredFeatures;
        int border = 20;
        for (const auto& pt : prevKeypointsCPU_) {
            if (pt.x > border && pt.y > border && 
                pt.x < (gray.cols - border) && pt.y < (gray.rows - border)) {
                filteredFeatures.push_back(pt);
            }
        }
        prevKeypointsCPU_ = filteredFeatures;
        
        if (prevKeypointsCPU_.size() < MIN_TRACKING_FEATURES) {
            logMessage("Warning: Only " + std::to_string(prevKeypointsCPU_.size()) + 
                      " features detected (minimum: " + std::to_string(MIN_TRACKING_FEATURES) + ")", true);
        }
    }

    void Stabilizer::generateTransform(const cv::Mat &frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        if (prevKeypointsCPU_.empty()) {
            detectInitialFeatures(gray);
            prevGrayCPU_ = gray.clone();
            
            // Add zero transform for first frame
            transforms_.push_back(cv::Vec3f(0, 0, 0));
            if (path_.empty()) {
                path_.push_back(cv::Vec3f(0, 0, 0));
            } else {
                path_.push_back(path_.back());
            }
            return;
        }
        
        // Track features using optical flow
        std::vector<cv::Point2f> currKeypoints;
        std::vector<uchar> status;
        std::vector<float> errors;
        
        cv::calcOpticalFlowPyrLK(
            prevGrayCPU_, gray,
            prevKeypointsCPU_, currKeypoints,
            status, errors,
            cv::Size(21, 21), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
        );
        
        // Filter good matches
        std::vector<cv::Point2f> goodPrev, goodCurr;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && errors[i] < 30.0f) {
                goodPrev.push_back(prevKeypointsCPU_[i]);
                goodCurr.push_back(currKeypoints[i]);
            }
        }
        
        // Check feature count and re-detect if needed
        if (goodPrev.size() < MIN_TRACKING_FEATURES) {
            logMessage("Low feature count (" + std::to_string(goodPrev.size()) + "), re-detecting");
            detectInitialFeatures(gray);
            prevGrayCPU_ = gray.clone();
            
            // Add zero transform when re-detecting
            transforms_.push_back(cv::Vec3f(0, 0, 0));
            if (path_.empty()) {
                path_.push_back(cv::Vec3f(0, 0, 0));
            } else {
                path_.push_back(path_.back());
            }
            return;
        }
        
        // Remove outliers
        removeOutliers(goodPrev, goodCurr);
        
        // Calculate transformation
        cv::Vec3f transform = calculateRigidTransform(goodPrev, goodCurr);
        
        // Apply shake suppression
        transform = suppressShake(transform);
        
        // Store transformation
        transforms_.push_back(transform);
        
        // Update cumulative path
        if (path_.empty()) {
            path_.push_back(transform);
        } else {
            cv::Vec3f newPos = path_.back() + transform;
            path_.push_back(newPos);
        }
        
        // Update for next frame
        prevKeypointsCPU_ = currKeypoints;
        prevGrayCPU_ = gray.clone();
        
        // Periodically re-detect features to maintain tracking quality
        static int framesSinceDetection = 0;
        if (++framesSinceDetection > 30) {
            detectInitialFeatures(gray);
            framesSinceDetection = 0;
        }
    }

    void Stabilizer::removeOutliers(std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &currPts) {
        if (prevPts.size() != currPts.size() || prevPts.size() < 4) {
            return;
        }
        
        // Calculate motion vectors
        std::vector<cv::Point2f> motions(prevPts.size());
        for (size_t i = 0; i < prevPts.size(); i++) {
            motions[i] = currPts[i] - prevPts[i];
        }
        
        // Calculate median motion for robust estimation
        std::vector<float> dx(motions.size()), dy(motions.size());
        for (size_t i = 0; i < motions.size(); i++) {
            dx[i] = motions[i].x;
            dy[i] = motions[i].y;
        }
        
        std::nth_element(dx.begin(), dx.begin() + dx.size()/2, dx.end());
        std::nth_element(dy.begin(), dy.begin() + dy.size()/2, dy.end());
        
        cv::Point2f medianMotion(dx[dx.size()/2], dy[dy.size()/2]);
        
        // Filter points based on distance from median
        std::vector<cv::Point2f> filteredPrev, filteredCurr;
        for (size_t i = 0; i < motions.size(); i++) {
            float dist = cv::norm(motions[i] - medianMotion);
            if (dist <= OUTLIER_THRESHOLD) {
                filteredPrev.push_back(prevPts[i]);
                filteredCurr.push_back(currPts[i]);
            }
        }
        
        // Only update if we have enough points remaining
        if (filteredPrev.size() >= 10) {
            prevPts = filteredPrev;
            currPts = filteredCurr;
        }
    }

    cv::Vec3f Stabilizer::calculateRigidTransform(const std::vector<cv::Point2f> &prevPts, 
                                                  const std::vector<cv::Point2f> &currPts) {
        if (prevPts.size() < 3) {
            return cv::Vec3f(0, 0, 0);
        }
        
        // Calculate centroids
        cv::Point2f prevCenter(0, 0), currCenter(0, 0);
        for (size_t i = 0; i < prevPts.size(); i++) {
            prevCenter += prevPts[i];
            currCenter += currPts[i];
        }
        prevCenter /= static_cast<float>(prevPts.size());
        currCenter /= static_cast<float>(currPts.size());
        
        // Translation
        float dx = currCenter.x - prevCenter.x;
        float dy = currCenter.y - prevCenter.y;
        
        // Rotation using cross-correlation
        float numerator = 0, denominator = 0;
        for (size_t i = 0; i < prevPts.size(); i++) {
            cv::Point2f p1 = prevPts[i] - prevCenter;
            cv::Point2f p2 = currPts[i] - currCenter;
            
            numerator += p1.x * p2.y - p1.y * p2.x;
            denominator += p1.x * p2.x + p1.y * p2.y;
        }
        
        float da = 0;
        if (std::abs(denominator) > 1e-6) {
            da = std::atan2(numerator, denominator);
        }
        
        return cv::Vec3f(dx, dy, da);
    }

    cv::Vec3f Stabilizer::suppressShake(const cv::Vec3f &transform) {
        float translationMag = std::sqrt(transform[0] * transform[0] + transform[1] * transform[1]);
        float rotationMag = std::abs(transform[2]);
        
        cv::Vec3f filtered = transform;
        
        // Detect and suppress shake
        if (translationMag < SHAKE_THRESHOLD_PX && rotationMag < ROTATION_SHAKE_RAD) {
            // This looks like camera shake - dampen it significantly
            filtered[0] *= SHAKE_DAMPING_FACTOR;
            filtered[1] *= SHAKE_DAMPING_FACTOR;
            filtered[2] *= SHAKE_DAMPING_FACTOR;
            
            logMessage("Shake suppressed (t=" + std::to_string(translationMag) + 
                      ", r=" + std::to_string(rotationMag * 180.0f / M_PI) + "°)");
        }
        
        return filtered;
    }

    cv::Mat Stabilizer::applyNextSmoothTransform() {
        if (frameQueue_.empty()) {
            return cv::Mat();
        }
        
        cv::Mat frame = frameQueue_.front();
        int frameIndex = frameIndexQueue_.front();
        frameQueue_.pop_front();
        frameIndexQueue_.pop_front();
        
        // Ensure we have valid indices
        if (frameIndex >= static_cast<int>(transforms_.size()) || 
            frameIndex >= static_cast<int>(path_.size())) {
            return frame;
        }
        
        // Apply box filter smoothing to path
        updateSmoothedPath();
        
        if (frameIndex >= static_cast<int>(smoothedPath_.size())) {
            return frame;
        }
        
        // Calculate stabilization transform
        cv::Vec3f rawPath = path_[frameIndex];
        cv::Vec3f smoothPath = smoothedPath_[frameIndex];
        cv::Vec3f correction = smoothPath - rawPath;
        
        // Apply transformation
        return applyTransform(frame, correction);
    }

    void Stabilizer::updateSmoothedPath() {
        if (path_.empty()) return;
        
        smoothedPath_.resize(path_.size());
        
        for (size_t i = 0; i < path_.size(); i++) {
            cv::Vec3f sum(0, 0, 0);
            int count = 0;
            
            // Define smoothing window
            int radius = static_cast<int>(boxKernel_.size()) / 2;
            int start = std::max(0, static_cast<int>(i) - radius);
            int end = std::min(static_cast<int>(path_.size() - 1), static_cast<int>(i) + radius);
            
            // Apply box filter
            for (int j = start; j <= end; j++) {
                sum += path_[j];
                count++;
            }
            
            smoothedPath_[i] = sum / static_cast<float>(count);
        }
    }

    cv::Mat Stabilizer::applyTransform(const cv::Mat &frame, const cv::Vec3f &transform) {
        // Create transformation matrix
        cv::Mat M = cv::Mat::eye(2, 3, CV_32F);
        
        float cosA = std::cos(transform[2]);
        float sinA = std::sin(transform[2]);
        
        M.at<float>(0, 0) = cosA;
        M.at<float>(0, 1) = -sinA;
        M.at<float>(1, 0) = sinA;
        M.at<float>(1, 1) = cosA;
        M.at<float>(0, 2) = transform[0];
        M.at<float>(1, 2) = transform[1];
        
        // Determine border mode
        int borderMode = cv::BORDER_REFLECT_101; // Default
        
        if (params_.borderType == "black") {
            borderMode = cv::BORDER_CONSTANT;
        } else if (params_.borderType == "replicate") {
            borderMode = cv::BORDER_REPLICATE;
        } else if (params_.borderType == "reflect") {
            borderMode = cv::BORDER_REFLECT;
        } else if (params_.borderType == "wrap") {
            borderMode = cv::BORDER_WRAP;
        }
        
        cv::Mat stabilized;
        
        // Create a larger canvas to show border effects when not cropping
        if (params_.cropNZoom == 0) {
            // Add border padding to make border effects visible
            int borderPadding = params_.borderSize;
            cv::Size largerSize(frame.cols + 2 * borderPadding, frame.rows + 2 * borderPadding);
            
            // Adjust transformation matrix to account for padding offset
            M.at<float>(0, 2) += borderPadding;
            M.at<float>(1, 2) += borderPadding;
            
            // Apply transformation with larger canvas
            cv::Mat largerCanvas;
            cv::warpAffine(frame, largerCanvas, M, largerSize, 
                          cv::INTER_LINEAR, borderMode);
            
            // Calculate crop region to maintain original size with visible borders
            int cropX = borderPadding / 2;
            int cropY = borderPadding / 2;
            
            // Ensure crop region is within bounds
            cropX = std::max(0, std::min(cropX, largerCanvas.cols - frame.cols));
            cropY = std::max(0, std::min(cropY, largerCanvas.rows - frame.rows));
            
            // Crop back to original size, showing border effects at edges
            cv::Rect cropRect(cropX, cropY, frame.cols, frame.rows);
            stabilized = largerCanvas(cropRect).clone();
            
            logMessage("Applied stabilization with borders (canvas: " + 
                      std::to_string(largerSize.width) + "x" + std::to_string(largerSize.height) + 
                      " -> output: " + std::to_string(stabilized.cols) + "x" + std::to_string(stabilized.rows) + ")");
        } else {
            // Original behavior for crop mode
            cv::warpAffine(frame, stabilized, M, frame.size(), 
                          cv::INTER_LINEAR, borderMode);
        }
        
        return stabilized;
    }

    // Simplified implementations of required functions
    std::vector<float> Stabilizer::boxFilterConvolve(const std::vector<float> &path) {
        if (path.empty()) return {};
        
        int radius = static_cast<int>(boxKernel_.size()) / 2;
        std::vector<float> result(path.size());
        
        for (size_t i = 0; i < path.size(); i++) {
            float sum = 0;
            int count = 0;
            
            int start = std::max(0, static_cast<int>(i) - radius);
            int end = std::min(static_cast<int>(path.size() - 1), static_cast<int>(i) + radius);
            
            for (int j = start; j <= end; j++) {
                sum += path[j];
                count++;
            }
            
            result[i] = sum / count;
        }
        
        return result;
    }

    // Minimal implementations for compatibility
    cv::Mat Stabilizer::convertColorAndEnhanceCPU(const cv::Mat &frameBGR) {
        cv::Mat gray;
        cv::cvtColor(frameBGR, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

    std::vector<cv::Point2f> Stabilizer::detectFeatures(const cv::Mat &grayFrame) {
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(grayFrame, corners, params_.maxCorners, 
                               params_.qualityLevel, params_.minDistance, 
                               cv::noArray(), params_.blockSize);
        return corners;
    }

    void Stabilizer::filterOutliers(std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &currPts) {
        removeOutliers(prevPts, currPts);
    }

    // Utility function implementations
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



} // namespace vs
