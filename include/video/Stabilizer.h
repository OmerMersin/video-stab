#ifndef VIDEO_STABILIZER_HPP
#define VIDEO_STABILIZER_HPP

// Compiler optimization hints for ARM processors (Jetson)
#ifdef __ARM_NEON
#define STABILIZER_SIMD_OPTIMIZED
#endif

#include <opencv2/opencv.hpp>

// For GPU: make sure you've built OpenCV with CUDA, e.g. HAVE_OPENCV_CUDAARITHM, HAVE_OPENCV_CUDAOPTFLOW, etc.
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

    // Professional stabilization structures
    struct Transform {
        float dx = 0.0f;
        float dy = 0.0f;
        float da = 0.0f;
        
        Transform() = default;
        Transform(float x, float y, float a) : dx(x), dy(y), da(a) {}
    };
    
    struct MotionSample {
        Transform transform;
        float magnitude = 0.0f;
        float confidence = 0.0f;
        int timestamp = 0;
    };
    
    enum class MotionType {
        NORMAL,
        INTENTIONAL_PAN,
        CAMERA_SHAKE,
        WALKING_VIBRATION,
        VEHICLE_VIBRATION
    };
    
    enum class MotionIntent {
        NORMAL,
        DELIBERATE_PAN,
        SHAKE_REMOVAL,
        FOLLOW_ACTION
    };
    
    enum class SceneType {
        NORMAL,
        SPORT,
        DRONE,
        HANDHELD,
        VEHICLE
    };

    /**
     * @brief A GPU-Accelerated Stabilizer replicating the logic from vidgear's stabilizer.py
     */
    class Stabilizer
    {
    public:
        /**
         * @brief Parameters for the stabilizer.
         */
        struct Parameters
        {
            bool useCuda = false;              ///< Use CUDA GPU acceleration if available
            bool logging = false;              ///< Enable console logging
            
            int smoothingRadius = 30;          ///< Radius for path smoothing (in frames)
            int maxCorners = 200;              ///< Maximum corners for feature detection
            double qualityLevel = 0.01;        ///< Quality level for corner detection
            double minDistance = 30.0;         ///< Minimum distance between corners
            int blockSize = 3;                 ///< Size of window for corner detection
            
            std::string borderType = "black";  ///< Border type: "black", "reflect", "replicate", "wrap"
            int borderSize = 0;                ///< Border size for stabilization (pixels)
            bool cropNZoom = false;           ///< Crop and zoom to remove black borders
            
            // Advanced parameters for professional stabilization
            std::string smoothingMethod = "box";  ///< Smoothing method: "box", "gaussian", "kalman"
            double gaussianSigma = 2.0;           ///< Sigma for Gaussian smoothing
            bool motionPrediction = true;         ///< Enable motion prediction
            bool horizonLock = false;             ///< Lock horizon for aerial/vehicle footage
            
            // Feature detection method
            enum FeatureDetector {
                GFTT,    ///< Good Features to Track (default)
                ORB,     ///< ORB features
                FAST,    ///< FAST corners
                BRISK    ///< BRISK features
            };
            
            FeatureDetector featureDetector = GFTT;  ///< Feature detection method
            int orbFeatures = 500;                   ///< Number of ORB features (if using ORB)
            int fastThreshold = 10;                  ///< FAST threshold (if using FAST)
            
            // ROI (Region of Interest) for feature detection
            bool useROI = false;              ///< Whether to use ROI for feature detection
            cv::Rect roi = cv::Rect();        ///< ROI rectangle (if useROI is true)
            
            // Advanced stabilization parameters
            bool adaptiveSmoothing = false;   ///< Enable adaptive smoothing based on motion
            int minSmoothingRadius = 5;       ///< Minimum smoothing radius for adaptive mode
            int maxSmoothingRadius = 50;      ///< Maximum smoothing radius for adaptive mode
            
            double outlierThreshold = 3.0;   ///< Threshold for outlier rejection (in standard deviations)
            double intentionalMotionThreshold = 20.0;  ///< Threshold for detecting intentional motion
            
            // Multi-stage smoothing parameters (VT3000 style)
            int stageOneRadius = 10;          ///< First stage smoothing radius
            int stageTwoRadius = 25;          ///< Second stage smoothing radius
            bool useTemporalFiltering = false; ///< Enable temporal filtering across multiple frames
            int temporalWindowSize = 5;      ///< Window size for temporal filtering
            
            // Fade border parameters
            float fadeAlpha = 0.1f;           ///< Alpha for fade border effect
            int fadeDuration = 30;            ///< Duration for fade effect (frames)
            
            // Motion analysis parameters
            float motionThresholdLow = 5.0f;  ///< Low motion threshold for dynamic borders
            float motionThresholdHigh = 20.0f; ///< High motion threshold for dynamic borders
            float borderScaleFactor = 2.0f;  ///< Factor for scaling border based on motion
            
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
            
            // Virtual Canvas Stabilization Parameters
            bool enableVirtualCanvas = false;       ///< Enable virtual canvas stabilization for jitter-free output
            float canvasScaleFactor = 1.5f;          ///< Canvas size multiplier (1.5 = 50% larger than frame)
            int temporalBufferSize = 30;            ///< Number of frames to keep for canvas filling
            float canvasBlendWeight = 0.7f;          ///< Blending weight for canvas temporal filling
            bool adaptiveCanvasSize = true;          ///< Dynamically adjust canvas size based on motion
            float maxCanvasScale = 2.0f;             ///< Maximum canvas scale factor
            float minCanvasScale = 1.2f;             ///< Minimum canvas scale factor
            bool preserveEdgeQuality = true;         ///< Use high-quality edge preservation for canvas
            int edgeBlendRadius = 20;                ///< Radius for edge blending between current and temporal frames
            
            // HF: Drone high-frequency vibration suppression parameters
            bool droneHighFreqMode = false;          ///< Enable drone prop vibration suppression mode
            float hfShakePx = 1.5f;                 ///< Micro-jitter amplitude threshold in analysis pixels
            int hfAnalysisMaxWidth = 960;           ///< Maximum analysis resolution width in drone mode
            float hfRotLPAlpha = 0.2f;              ///< Low-pass alpha for rotation smoothing (0.0-1.0)
            bool enableConditionalCLAHE = true;     ///< Re-enable CLAHE when feature starvation detected
            
            // HF: Dead zone parameters for freeze shot
            float hfDeadZoneThreshold = 2.0f;       ///< Motion threshold below which camera freezes completely
            int hfFreezeDuration = 10;              ///< Number of frames to maintain freeze after entering dead zone
            float hfMotionAccumulatorDecay = 0.9f;  ///< How quickly accumulated motion decays
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

        // Color conversion and enhancement
        cv::Mat convertColorAndEnhanceCPU(const cv::Mat &frameBGR);
    #ifdef HAVE_OPENCV_CUDAARITHM
        cv::cuda::GpuMat convertColorAndEnhanceGPU(const cv::Mat &frameBGR);
    #endif

        // Feature detection methods
        std::vector<cv::Point2f> detectFeatures(const cv::Mat &grayFrame);
    #ifdef HAVE_OPENCV_CUDAFEATURES2D
        cv::cuda::GpuMat detectFeaturesGPU(const cv::cuda::GpuMat &grayFrameGPU);
    #endif
        
        // Outlier filtering
        void filterOutliers(std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &currPts);
        
        // Path smoothing methods
        std::vector<float> gaussianFilterConvolve(const std::vector<float> &path, float sigma);
        std::vector<float> kalmanFilterSmooth(const std::vector<float> &path);
        void adaptSmoothingRadius(const cv::Vec3f &recentMotion);
        
        // Motion analysis
        bool isIntentionalMotion(const cv::Vec3f &motion);
        cv::Vec3f predictNextMotion();
        
        // ROI calculation
        cv::Rect calculateROI(const cv::Mat &frame);
        
        // Advanced adaptive parameters
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

        // Professional stabilization methods
        void initializeRobustStabilization();
        Transform validateAndFilterMotion(const Transform& rawTransform);
        float calculateMotionConfidence(const Transform& transform);
        MotionType classifyMotion(const Transform& transform, float magnitude);
        Transform applyIntentionalMotionFilter(const Transform& transform);
        Transform applyShakeRemovalFilter(const Transform& transform);
        Transform applyWalkingStabilization(const Transform& transform);
        Transform applyVehicleStabilization(const Transform& transform);
        Transform applyGeneralStabilization(const Transform& transform);
        Transform applyHorizonLock(const Transform& transform);
        void updatePredictiveFilters(const Transform& transform);
        
        // Advanced motion analysis methods
        int calculateAdaptiveRadius(const std::vector<float>& px, const std::vector<float>& py, const std::vector<float>& pa);
        MotionIntent analyzeMotionIntent(const cv::Vec3f& motion, int frameIndex);
        float calculateAdaptiveStabilizationStrength(MotionIntent intent, const cv::Vec3f& motion);
        float calculateVariance(const std::vector<float>& values);
        float calculateConsistency(const std::vector<float>& values);
        
        // Virtual Canvas Stabilization Methods
        cv::Mat applyRealtimeVirtualCanvas(const cv::Mat& currentFrame, const cv::Vec3f& transform);
        void fillCanvasBackground();
        cv::Mat applyVirtualCanvasStabilization(const cv::Mat& currentFrame, const cv::Vec3f& transform);
        void updateTemporalFrameBuffer(const cv::Mat& frame, const cv::Vec3f& transform);
        cv::Mat createVirtualCanvas(const cv::Mat& currentFrame, const cv::Vec3f& transform);
        cv::Mat blendTemporalRegions(const cv::Mat& canvas, const cv::Mat& currentFrame, const cv::Vec3f& transform);
        float calculateOptimalCanvasSize(const cv::Vec3f& recentMotion);
        cv::Mat extractTemporalRegion(const cv::Mat& frame, const cv::Rect& region, const cv::Vec3f& frameTransform);
        void seamlessBlend(cv::Mat& target, const cv::Mat& source, const cv::Rect& region, float weight);
        bool isRegionAvailable(const cv::Rect& region, const cv::Vec3f& transform, int frameIndex);
        cv::Mat applyMotionCompensation(const cv::Mat& temporalFrame, const cv::Vec3f& motionVector);
        
        // HF: Drone high-frequency vibration suppression methods
        cv::Size calculateDroneAnalysisSize(const cv::Mat& frame);
        Transform applyMicroShakeSuppression(const Transform& rawTransform);
        Transform applyRotationLowPass(const Transform& transform);
        void updateTranslationHistory(const cv::Vec2f& translation);
        cv::Vec2f calculateMedianTranslation();
        bool shouldApplyConditionalCLAHE(int detectedFeatureCount);
        cv::Mat applyConditionalCLAHE(const cv::Mat& grayFrame);
        float mapJitterFrequencyToCutoff(Parameters::JitterFrequency freq);
        
        // HF: Dead zone freeze shot methods
        Transform applyDeadZoneFreeze(const Transform& rawTransform);
        bool shouldEnterDeadZone(const Transform& transform);
        void updateMotionAccumulator(const Transform& transform);
        Transform getFrozenTransform();

        // Core shake-avoiding stabilizer methods
        cv::Mat initializeFirstFrame(const cv::Mat &frame);
        void detectInitialFeatures(const cv::Mat &gray);
        void removeOutliers(std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &currPts);
        cv::Vec3f calculateRigidTransform(const std::vector<cv::Point2f> &prevPts, 
                                         const std::vector<cv::Point2f> &currPts);
        cv::Vec3f suppressShake(const cv::Vec3f &transform);
        void updateSmoothedPath();
        cv::Mat applyTransform(const cv::Mat &frame, const cv::Vec3f &transform);

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

        // For CPU-based CLAHE
        cv::Ptr<cv::CLAHE> claheCPU_;

    #if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
        // GPU-based CLAHE
        cv::Ptr<cv::CLAHE> claheGPU_;
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

        // Additional GPU streams for parallel processing
    #if defined(HAVE_OPENCV_CUDAARITHM) || defined(HAVE_OPENCV_CUDAWARPING)
        cv::cuda::Stream analysisStream_;
        cv::cuda::Stream warpStream_;
        
        // Pre-allocated GPU buffers for zero-copy operations
        cv::cuda::GpuMat gpuFrameBuffer_;
        cv::cuda::GpuMat gpuGrayBuffer_;
        cv::cuda::GpuMat gpuPrevGrayBuffer_;
        cv::cuda::GpuMat gpuWarpBuffer_;
    #endif

        // Professional stabilization member variables
        std::vector<MotionSample> motionHistory_;
        std::vector<float> intentionalMotionBuffer_;
        std::vector<float> velocityFilter_;
        std::vector<float> accelerationFilter_;
        
        bool horizonLockEnabled_ = true;
        float horizonAngle_ = 0.0f;
        float horizonConfidence_ = 0.0f;
        
        float intentionalMotionThreshold_ = 15.0f;
        float shakeLevelThreshold_ = 2.0f;
        float minSmoothingStrength_ = 0.3f;
        float maxSmoothingStrength_ = 0.95f;
        
        SceneType lastSceneType_ = SceneType::NORMAL;
        int sceneChangeCounter_ = 0;
        int frameCount_ = 0;
        
        Transform lastValidTransform_;
        float motionQuality_ = 0.8f;
        float stabilizationStrength_ = 0.7f;
        
        // Virtual Canvas Stabilization Members
        std::deque<cv::Mat> temporalFrameBuffer_;           ///< Buffer of recent frames for canvas filling
        std::deque<cv::Vec3f> temporalTransformBuffer_;     ///< Corresponding transforms for temporal frames
        cv::Mat virtualCanvas_;                             ///< The larger canvas for stabilization
        cv::Size canvasSize_;                              ///< Current canvas size
        cv::Point2f canvasCenter_;                         ///< Center point of the canvas
        cv::Mat canvasBlendMask_;                          ///< Blending mask for seamless integration
        float currentCanvasScale_ = 1.5f;                  ///< Current dynamic canvas scale
        int virtualCanvasFrameCount_ = 0;                  ///< Frame counter for canvas operations
        
        // HF: Drone high-frequency vibration suppression members
        std::deque<cv::Vec2f> hfTranslationHistory_;       ///< Recent translation history for median reference
        cv::Vec2f hfMedianTranslation_;                    ///< Current median translation reference
        float hfRotationLowPass_ = 0.0f;                   ///< Low-pass filtered rotation value
        int hfFeatureStarvationCounter_ = 0;               ///< Counter for feature starvation detection
        cv::Ptr<cv::CLAHE> hfConditionalCLAHE_;           ///< Conditional CLAHE for feature enhancement
        
        // HF: Dead zone freeze shot state
        bool hfInDeadZone_ = false;                        ///< Currently in dead zone (frozen)
        int hfFreezeCounter_ = 0;                          ///< Frames remaining in freeze
        float hfMotionAccumulator_ = 0.0f;                 ///< Accumulated motion magnitude
        Transform hfFrozenTransform_;                      ///< Last transform when entering freeze mode
        
    private:
        // Implementation details
        std::vector<float> boxFilterConvolve(const std::vector<float> &path);

        // Jetson Orin Nano specific optimizations
        void optimizeForJetson();
        void processFrameInPlace(cv::Mat& frame);
        cv::Mat fastApproxTransform(const cv::Mat& frame, const cv::Vec3f& transform);
    };

}

#endif // VIDEO_STABILIZER_HPP
