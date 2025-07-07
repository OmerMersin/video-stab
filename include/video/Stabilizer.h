#ifndef VIDEO_STABILIZER_HPP
#define VIDEO_STABILIZER_HPP

// Compiler optimization hints for ARM processors (Jetson)
#ifdef __ARM_NEON
#define STABILIZER_SIMD_OPTIMIZED
#endif

#include <opencv2/opencv.hpp>

// For GPU: make sure you’ve built OpenCV with CUDA, e.g. HAVE_OPENCV_CUDAARITHM, HAVE_OPENCV_CUDAOPTFLOW, etc.
#if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
#include <opencv2/core/cuda.hpp>     // for cv::cuda::Stream
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
            std::string borderType = "black";///< "black", "reflect", "replicate", "fade", etc. 
            int  borderSize = 0;            ///< Additional border
            bool cropNZoom = false;         ///< If true, crop away borders, then zoom to original size
            bool logging = false;           ///< Enable/disable debug logs
            
            // Fade border parameters (when borderType = "fade")
            int fadeDuration = 30;          ///< Number of frames over which to apply the fade effect
            float fadeAlpha = 0.9;          ///< Rate of decay for the fade effect (0-1)

            bool useCuda = true;            ///< Toggle GPU usage for color conv, feature detection, optical flow, warp
            int  maxCorners = 200;          ///< GFTT param
            double qualityLevel = 0.05;     
            double minDistance = 30.0;      
            int  blockSize = 3;

            // SightLine VT3000-inspired parameters
            bool adaptiveSmoothing = false;  ///< Adapt smoothing radius based on motion
            int minSmoothingRadius = 10;     ///< Minimum radius when adaptiveSmoothing is true
            int maxSmoothingRadius = 50;     ///< Maximum radius when adaptiveSmoothing is true
            
            bool outlierRejection = true;    ///< Filter out motion outliers
            double outlierThreshold = 3.0;   ///< Threshold for outlier rejection (in standard deviations)
            
            std::string smoothingMethod = "gaussian"; ///< "box", "gaussian", "kalman", or "deep"
            double gaussianSigma = 15.0;     ///< Sigma for Gaussian smoothing
            
            bool motionPrediction = false;   ///< Use motion prediction for stabilization
            double intentionalMotionThreshold = 0.7; ///< Threshold to distinguish between intentional and unintentional motion
            
            bool useROI = false;             ///< Use region of interest for feature detection
            cv::Rect roi = cv::Rect(0, 0, 0, 0); ///< Region of interest (if useROI is true)
            
            bool horizonLock = false;        ///< Lock the horizon (prevent rotation)
            
            enum FeatureDetector {
                GFTT,       ///< Good Features To Track (default)
                ORB,        ///< ORB features
                FAST,       ///< FAST features
                BRISK       ///< BRISK features
            };
            
            FeatureDetector featureDetector = GFTT; ///< Feature detector to use
            int fastThreshold = 20;           ///< Threshold for FAST feature detector
            int orbFeatures = 500;            ///< Number of features for ORB detector
            
            // VT3000-specific advanced features
            bool multiStageSmoothing = true;   ///< Use multi-stage smoothing pipeline (VT3000 style)
            bool dynamicBorderScaling = true;  ///< Dynamically scale borders based on motion magnitude
            double borderScaleFactor = 1.5;    ///< Scale factor for dynamic borders
            
            int stageOneRadius = 30;           ///< First stage smoothing radius
            int stageTwoRadius = 60;           ///< Second stage smoothing radius
            
            bool featureWeighting = true;      ///< Weight features by reliability score
            bool sceneClassification = true;   ///< Classify scene type to adjust parameters
            
            double motionThresholdLow = 5.0;   ///< Low motion threshold (pixels)
            double motionThresholdHigh = 50.0; ///< High motion threshold (pixels)
            
            bool useTemporalFiltering = true;  ///< Use temporal filtering across frames
            int temporalWindowSize = 5;        ///< Window size for temporal filtering
            
            bool rollCompensation = true;      ///< Compensate for roll/banking during turns
            double rollCompensationFactor = 0.75; ///< How much to compensate for roll (0-1)
            
            bool deepStabilization = false;    ///< Use deep learning based stabilization (requires model)
            std::string modelPath = "";        ///< Path to deep stabilization model
            
            enum JitterFrequency {
                LOW,      ///< Low frequency jitter (slow oscillations)
                MEDIUM,   ///< Medium frequency jitter
                HIGH,     ///< High frequency jitter (vibrations)
                ADAPTIVE  ///< Adaptive multi-frequency filtering
            };
            
            JitterFrequency jitterFrequency = ADAPTIVE;  ///< Target jitter frequency
            bool separateTranslationRotation = true;    ///< Handle translation and rotation separately
            bool useImuData = false;                    ///< Incorporate IMU data if available

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
        
        // SightLine-inspired methods
        std::vector<cv::Point2f> detectFeatures(const cv::Mat &grayFrame);
        cv::cuda::GpuMat detectFeaturesGPU(const cv::cuda::GpuMat &grayFrameGPU);
        void filterOutliers(std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &currPts);
        std::vector<float> gaussianFilterConvolve(const std::vector<float> &path, float sigma);
        std::vector<float> kalmanFilterSmooth(const std::vector<float> &path);
        void adaptSmoothingRadius(const cv::Vec3f &recentMotion);
        bool isIntentionalMotion(const cv::Vec3f &motion);
        cv::Vec3f predictNextMotion();
        cv::Rect calculateROI(const cv::Mat &frame);
        void updateAdaptiveParameters();

        // VT3000-inspired advanced methods
        void applyMultiStageSmoothing(std::vector<float> &x, std::vector<float> &y, std::vector<float> &a);
        std::vector<float> temporalFilter(const std::vector<float> &path);
        void classifySceneType(const cv::Mat &frame);
        void adjustParametersForScene();
        float calculateDynamicBorderSize(const std::vector<cv::Vec3f> &recentTransforms);
        cv::Mat applyDeepStabilization(const cv::Mat &frame);
        void compensateForRoll(cv::Vec3f &tform);
        std::vector<float> adaptiveFrequencyFilter(const std::vector<float> &path);
        std::vector<float> butterworthFilter(const std::vector<float> &path, double cutoffFreq, int order);
        void separateMotionComponents(const cv::Vec3f &motion, cv::Vec2f &translation, float &rotation);
        float getFeatureReliabilityScore(const std::vector<cv::Point2f> &prevPts, const std::vector<cv::Point2f> &currPts);

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

    #if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
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

        // For fade border type
        cv::Mat borderHistory_;       // Stores the background for fading
        int fadeFrameCount_ = 0;      // Counter for fade effect
        
        // Basic state
        int frameWidth_  = 0;
        int frameHeight_ = 0;
        bool firstFrame_ = true;
        int  nextFrameIndex_ = 0;
        cv::Size origSize_; // For crop+zoom

        // Parameters
        Parameters params_;
       
        int analysisWidth_{0};
	int analysisHeight_{0};
	
	

    private:
        // Implementation details
        std::vector<float> boxFilterConvolve(const std::vector<float> &path);
    };

}

#endif // VIDEO_STABILIZER_HPP
