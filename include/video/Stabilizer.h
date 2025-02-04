#ifndef VIDEO_STABILIZER_HPP
#define VIDEO_STABILIZER_HPP

#include <opencv2/opencv.hpp>

// For GPU: make sure you’ve built OpenCV with CUDA, e.g. HAVE_OPENCV_CUDAARITHM, HAVE_OPENCV_CUDAOPTFLOW, etc.
#ifdef HAVE_OPENCV_CUDAARITHM
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#endif

#include <deque>
#include <string>
#include <vector>

namespace vs {
     /**
     * @brief A GPU-Accelerated Stabilizer replicating the logic from vidgear’s stabilizer.py
     *        but using CUDA for color conversion, feature detection, optical flow, and warping.
     *
     *        - If useCuda == false, it falls back to the CPU approach.
     *        - If useCuda == true, all steps except estimateAffinePartial2D occur on GPU.
     *
     */
    class Stabilizer
    {
    public:
        struct Parameters {
            int  smoothingRadius = 25;       ///< Box filter radius for smoothing
            std::string borderType = "black";///< "black", "reflect", etc. 
            int  borderSize = 0;            ///< Additional border
            bool cropNZoom = false;         ///< If true, crop away borders, then zoom to original size
            bool logging = false;           ///< Enable/disable debug logs

            bool useCuda = true;            ///< Toggle GPU usage for color conv, feature detection, optical flow, warp
            int  maxCorners = 200;          ///< GFTT param
            double qualityLevel = 0.05;     
            double minDistance = 30.0;      
            int  blockSize = 3;             
        };

        explicit Stabilizer(const Parameters &params);
        ~Stabilizer();

        /**
         * @brief Pushes a new frame into the stabilizer. Returns a stabilized frame 
         *        if enough frames have accumulated (>= smoothingRadius), else empty.
         *
         * @param frame BGR frame (CPU cv::Mat)
         * @return Stabilized BGR frame (CPU cv::Mat) or empty cv::Mat if not ready.
         */
        cv::Mat stabilize(const cv::Mat &frame);

        /**
         * @brief Flush remaining frames in queue (after the stream ends).
         * @return Next stabilized frame or empty if none remain.
         */
        cv::Mat flush();

        /**
         * @brief Clear internal state. 
         */
        void clean();

    private:
        // Helper: to log messages if logging enabled
        void logMessage(const std::string &msg, bool isError=false) const;
        
        // Core steps
        void generateTransform(const cv::Mat &currFrameBGR);
        cv::Mat applyNextSmoothTransform();

        // GPU or CPU conversions
        cv::Mat convertColorAndEnhanceCPU(const cv::Mat &frameBGR);
    #ifdef HAVE_OPENCV_CUDAARITHM
        // Return a GPU Gray image with CLAHE applied
        cv::cuda::GpuMat convertColorAndEnhanceGPU(const cv::Mat &frameBGR);
    #endif

        // We store CPU frames in a queue, plus an index to match transforms
        std::deque<cv::Mat> frameQueue_;
        std::deque<int>     frameIndexQueue_;

        // Path data
        std::vector<cv::Vec3f> transforms_;   // (dx, dy, da) from frame i -> i+1
        std::vector<cv::Vec3f> path_;         // cumulative sum of transforms
        std::vector<cv::Vec3f> smoothedPath_; // path after box filter

        // Box filter kernel
        std::vector<float> boxKernel_;

        // Maintain "previous" GPU or CPU gray
        // We'll keep them in CPU if !useCuda, or in GpuMat if useCuda
        cv::Mat prevGrayCPU_;
    #ifdef HAVE_OPENCV_CUDAARITHM
        cv::cuda::GpuMat prevGrayGPU_;        // GPU version
    #endif
        bool useGpu_ = false;                 // store from params_.useCuda

        // Previous keypoints
        std::vector<cv::Point2f> prevKeypointsCPU_;
    #ifdef HAVE_OPENCV_CUDAOPTFLOW
        cv::cuda::GpuMat prevPtsGPU_;  // Nx1x2 32FC1 or Nx1x2 32FC2 for SparsePyrLK
    #endif

        // For CPU-based GFTT
        cv::Ptr<cv::CLAHE> claheCPU_;

    #ifdef HAVE_OPENCV_CUDAARITHM
        // GPU-based CLAHE
        cv::Ptr<cv::cuda::CLAHE> claheGPU_;
        cv::cuda::Stream cudaStream_;
    #endif

        // GPU-based detectors/optical flow if requested
    #ifdef HAVE_OPENCV_CUDAOPTFLOW
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> pyrLK_; 
    #endif
    #ifdef HAVE_OPENCV_CUDAFEATURES2D
        cv::Ptr<cv::cuda::CornersDetector> gfttDetector_;
    #endif

        // Border/wrapping
        int borderMode_ = cv::BORDER_CONSTANT;

        // Basic state
        int frameWidth_  = 0;
        int frameHeight_ = 0;
        bool firstFrame_ = true;
        int  nextFrameIndex_ = 0;
        cv::Size origSize_; // For crop+zoom

        // Parameters
        Parameters params_;

    private:
        // Implementation details
        std::vector<float> boxFilterConvolve(const std::vector<float> &path);
    };
}

#endif // VIDEO_STABILIZER_HPP
