#ifndef ROLL_CORRECTION_H
#define ROLL_CORRECTION_H

#include <opencv2/opencv.hpp>

namespace vs {

/**
 * @brief A simple GPU-based roll correction class
 */
class RollCorrection {
public:
    /**
     * @brief Configuration parameters for roll correction
     */
    struct Parameters {
        // Downscale factor for edge detection (0..1)
        double scaleFactor        = 0.25;

        // Canny thresholds (GPU)
        double cannyThresholdLow  = 50.0;
        double cannyThresholdHigh = 150.0;
        int    cannyAperture      = 3;   // Aperture for Sobel

        // Hough lines (GPU)
        float  houghRho           = 1.0f;
        float  houghTheta         = static_cast<float>(CV_PI / 180.0f);
        int    houghThreshold     = 100;

        // Acceptable angles for lines (in degrees, around horizontal)
        double angleFilterMin     = -10.0;
        double angleFilterMax     =  10.0;

        // Smoothing
        double angleSmoothingAlpha = 0.1;  // Exponential smoothing factor [0..1]
        double angleDecay          = 0.995; // Drift factor if no lines found (closer to 1 => slower drift)
        double maxAngleChangeDeg   = 0.5;   // Limit degrees of change per frame (0 => no clamp)
    };

    /**
     * @brief Detect and correct roll in an image using CUDA.
     *        Maintains internal static state for smoothing between frames.
     *
     * @param input   BGR image (cv::Mat)
     * @param params  Roll correction parameters
     * @return        Roll-corrected BGR image
     */
    static cv::Mat autoCorrectRoll(const cv::Mat& input, const Parameters& params);
};

} // namespace vs

#endif // ROLL_CORRECTION_H
