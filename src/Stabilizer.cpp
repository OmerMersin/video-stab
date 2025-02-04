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
#include "video/Stabilizer.h"

// If your OpenCV is built with these definitions, you can enable full GPU pipeline:
#ifdef HAVE_OPENCV_CUDAARITHM
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
        logMessage("Initializing...");

        useGpu_ = params_.useCuda;
        borderMode_ = mapBorderMode(params_.borderType);

        if(params_.cropNZoom && params_.borderType != "black") {
            // force black if crop+zoom
            logMessage("cropNZoom => ignoring borderType, using black.", false);
            borderMode_ = cv::BORDER_CONSTANT;
        }

        // 1D box kernel
        boxKernel_.resize(params_.smoothingRadius, 1.0f / params_.smoothingRadius);

        // CPU CLAHE
        claheCPU_ = cv::createCLAHE(2.0, cv::Size(8,8));

    #ifdef HAVE_OPENCV_CUDAARITHM
        if(useGpu_) {
            // GPU CLAHE
            claheGPU_ = cv::cuda::createCLAHE(2.0, cv::Size(8,8));
        }
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

            // Convert to gray
            if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
                prevGrayGPU_ = convertColorAndEnhanceGPU(frame);
    #endif
            } else {
                prevGrayCPU_ = convertColorAndEnhanceCPU(frame);
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

        // subsequent frames
        frameQueue_.push_back(frame.clone());
        frameIndexQueue_.push_back(nextFrameIndex_);

        generateTransform(frame);

        if(frameIndexQueue_.size() < (size_t)params_.smoothingRadius) {
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
        // Convert current frame to Gray
        cv::Mat currGrayCPU;
    #ifdef HAVE_OPENCV_CUDAARITHM
        cv::cuda::GpuMat currGrayGPU;
    #endif

        if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
            currGrayGPU = convertColorAndEnhanceGPU(currFrameBGR);
    #endif
        } else {
            currGrayCPU = convertColorAndEnhanceCPU(currFrameBGR);
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
                // estimate
                cv::Mat T = cv::Mat::eye(2,3,CV_64F);
                if(!validPrev.empty() && !validCurr.empty()) {
                    cv::Mat affine = cv::estimateAffinePartial2D(validPrev, validCurr);
                    if(!affine.empty()) {
                        T = affine;
                    }
                }
                double dx = T.at<double>(0,2);
                double dy = T.at<double>(1,2);
                double da = std::atan2(T.at<double>(1,0), T.at<double>(0,0));
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
                cv::Mat T = cv::Mat::eye(2,3,CV_64F);
                if(!validPrev.empty() && !validCurr.empty()) {
                    cv::Mat affine = cv::estimateAffinePartial2D(validPrev, validCurr);
                    if(!affine.empty()) {
                        T = affine;
                    }
                }
                double dx = T.at<double>(0,2);
                double dy = T.at<double>(1,2);
                double da = std::atan2(T.at<double>(1,0), T.at<double>(0,0));
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

        // Now detect GFTT on current for next iteration
        if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAARITHM
            // If we want new GFTT:
            cv::cuda::GpuMat currGray = currGrayGPU;
    #ifdef HAVE_OPENCV_CUDAFEATURES2D
            if(gfttDetector_) {
                cv::cuda::GpuMat cornersGPU;
                gfttDetector_->detect(currGray, cornersGPU, cv::noArray());
                // reshape to 1xN
                if(!cornersGPU.empty()) {
                    cornersGPU = cornersGPU.reshape(2, 1);
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
            // else fallback to CPU if you like...
    #endif
    #endif
        }
        else {
            // CPU GFTT
            std::vector<cv::Point2f> corners;
            cv::goodFeaturesToTrack(
                currGrayCPU, corners,
                params_.maxCorners,
                params_.qualityLevel,
                params_.minDistance,
                cv::noArray(),
                params_.blockSize
            );
            prevKeypointsCPU_ = corners;
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

        // Box filter path => smoothedPath
        std::vector<float> px, py, pa;
        px.reserve(path_.size());
        py.reserve(path_.size());
        pa.reserve(path_.size());
        for(const auto &v : path_) {
            px.push_back(v[0]);
            py.push_back(v[1]);
            pa.push_back(v[2]);
        }
        std::vector<float> sx = boxFilterConvolve(px);
        std::vector<float> sy = boxFilterConvolve(py);
        std::vector<float> sa = boxFilterConvolve(pa);

        smoothedPath_.resize(path_.size());
        for(size_t i=0; i<path_.size(); i++) {
            smoothedPath_[i] = cv::Vec3f(sx[i], sy[i], sa[i]);
        }

        cv::Vec3f raw = transforms_[oldestIdx];
        cv::Vec3f diff = smoothedPath_[oldestIdx] - path_[oldestIdx];
        cv::Vec3f tSmooth = raw + diff;

        float dx = tSmooth[0];
        float dy = tSmooth[1];
        float da = tSmooth[2];

        // 2x3
        cv::Mat T(2,3,CV_32F);
        T.at<float>(0,0) =  std::cos(da);
        T.at<float>(0,1) = -std::sin(da);
        T.at<float>(1,0) =  std::sin(da);
        T.at<float>(1,1) =  std::cos(da);
        T.at<float>(0,2) =  dx;
        T.at<float>(1,2) =  dy;

        // border
        cv::Mat frameWithBorder;
        if(params_.borderSize>0 && !params_.cropNZoom) {
            cv::copyMakeBorder(
                oldestFrame, frameWithBorder,
                params_.borderSize, params_.borderSize,
                params_.borderSize, params_.borderSize,
                borderMode_, cv::Scalar(0,0,0)
            );
        } else {
            frameWithBorder = oldestFrame;
        }

        // warp
        cv::Mat stabilized;
        if(useGpu_) {
    #ifdef HAVE_OPENCV_CUDAWARPING
            cv::cuda::GpuMat gpuIn, gpuOut;
            gpuIn.upload(frameWithBorder, cudaStream_);
            cv::cuda::warpAffine(
                gpuIn, gpuOut, T, gpuIn.size(),
                cv::INTER_LINEAR, borderMode_, cv::Scalar(), cudaStream_
            );
            gpuOut.download(stabilized, cudaStream_);
            cudaStream_.waitForCompletion();
    #else
            // fallback CPU
            cv::warpAffine(
                frameWithBorder, stabilized,
                T, frameWithBorder.size(),
                cv::INTER_LINEAR, borderMode_
            );
    #endif
        } else {
            cv::warpAffine(
                frameWithBorder, stabilized,
                T, frameWithBorder.size(),
                cv::INTER_LINEAR, borderMode_
            );
        }

        if(params_.cropNZoom && params_.borderSize>0) {
            int b = params_.borderSize;
            cv::Rect roi(b,b, stabilized.cols - 2*b, stabilized.rows - 2*b);
            roi &= cv::Rect(0,0, stabilized.cols, stabilized.rows);
            cv::Mat cropped = stabilized(roi).clone();
            if(!origSize_.empty()) {
                cv::resize(cropped, cropped, origSize_);
            }
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
        int r = params_.smoothingRadius;

        // approximate median
        std::vector<float> tmp = path;
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
        float med = tmp[tmp.size()/2];

        std::vector<float> padded(path.size()+2*r, med);
        for(size_t i=0; i<path.size(); i++){
            padded[r+i] = path[i];
        }

        std::vector<float> result(path.size());
        for(size_t i=0; i<path.size(); i++) {
            double sum=0.0;
            for(int k=0; k<r; k++){
                sum += padded[i+k];
            }
            result[i] = static_cast<float>(sum / r);
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
}