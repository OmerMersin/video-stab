#include "video/RollCorrection.h"

// GPU includes
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaobjdetect.hpp>   // For cv::cuda::HoughLinesDetector

namespace vs {

// Static variables to preserve angle across frames
static bool   sFirstFrame    = true;
static double sSmoothedAngle = 0.0;

cv::Mat RollCorrection::autoCorrectRoll(const cv::Mat& input, const Parameters& params)
{
    /***********************************************************************
     * 1. EARLY-OUT CHECKS
     ***********************************************************************/
    if (input.empty()) {
        return cv::Mat(); // Return empty if input is empty
    }
    if (sFirstFrame) {
        sFirstFrame = false;
        sSmoothedAngle = 0.0; // Reset angle on first usage
    }

    /***********************************************************************
     * 2. UPLOAD & DOWNSCALE
     ***********************************************************************/
    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(input);  // Send input to GPU

    cv::Size smallSize(
        static_cast<int>(input.cols * params.scaleFactor),
        static_cast<int>(input.rows * params.scaleFactor)
    );
    cv::cuda::GpuMat gpuSmall;
    if (smallSize.width > 0 && smallSize.height > 0) {
        cv::cuda::resize(gpuInput, gpuSmall, smallSize, 0, 0, cv::INTER_LINEAR);
    } else {
        // scaleFactor might be 1 or 0? Edge case => skip
        gpuSmall = gpuInput;
    }

    /***********************************************************************
     * 3. GRAYSCALE & CANNY (GPU)
     ***********************************************************************/
    cv::cuda::GpuMat gpuGray, gpuEdges;
    cv::cuda::cvtColor(gpuSmall, gpuGray, cv::COLOR_BGR2GRAY);

    // Create a local CannyEdgeDetector
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = 
        cv::cuda::createCannyEdgeDetector(
            params.cannyThresholdLow,
            params.cannyThresholdHigh,
            params.cannyAperture,
            false // L2 gradient
        );
    canny->detect(gpuGray, gpuEdges);

    /***********************************************************************
     * 4. HOUGH LINES (GPU)
     ***********************************************************************/
    cv::Ptr<cv::cuda::HoughLinesDetector> houghDetector = 
        cv::cuda::createHoughLinesDetector(
            params.houghRho, 
            params.houghTheta,
            params.houghThreshold
        );
    cv::cuda::GpuMat gpuLines;
    houghDetector->detect(gpuEdges, gpuLines);

    // If no lines, drift angle & return
    if (gpuLines.empty()) {
        sSmoothedAngle *= params.angleDecay; // slow drift to zero
        // Rotate using updated angle
        cv::Point2f center(input.cols / 2.0f, input.rows / 2.0f);
        cv::Mat rot = cv::getRotationMatrix2D(center, sSmoothedAngle, 1.0);

        cv::cuda::GpuMat gpuMapX, gpuMapY, gpuRotated;
        cv::cuda::buildWarpAffineMaps(rot, false, input.size(), gpuMapX, gpuMapY);
        cv::cuda::remap(gpuInput, gpuRotated, gpuMapX, gpuMapY, 
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        cv::Mat corrected;
        gpuRotated.download(corrected);
        return corrected;
    }

    /***********************************************************************
     * 5. DOWNLOAD LINES & COMPUTE AVERAGE ANGLE
     ***********************************************************************/
    cv::Mat linesMat;
    gpuLines.download(linesMat);  // Nx1 or 1xN vector of Vec2f

    std::vector<cv::Vec2f> lines;
    if (!linesMat.empty()) {
        lines.assign(
            linesMat.ptr<cv::Vec2f>(),
            linesMat.ptr<cv::Vec2f>() + linesMat.total()
        );
    }

    double sumAngle = 0.0;
    int    count    = 0;

    for (const auto& ln : lines) {
        float rho   = ln[0];
        float theta = ln[1];
        // Convert to degrees around horizontal [-90..+90]
        double angleDeg = (theta * 180.0 / CV_PI) - 90.0;
        // Filter
        if (angleDeg >= params.angleFilterMin && angleDeg <= params.angleFilterMax) {
            sumAngle += angleDeg;
            ++count;
        }
    }

    // If still no valid lines, drift
    if (count == 0) {
        sSmoothedAngle *= params.angleDecay;
    } else {
        double detectedAngle = sumAngle / count;
        // Exponential smoothing
        double newAngle = params.angleSmoothingAlpha * detectedAngle 
                        + (1.0 - params.angleSmoothingAlpha) * sSmoothedAngle;
        // Optional clamp
        double diff = newAngle - sSmoothedAngle;
        if (std::fabs(diff) > params.maxAngleChangeDeg && params.maxAngleChangeDeg > 0.0) {
            diff = (diff > 0) ? params.maxAngleChangeDeg : -params.maxAngleChangeDeg;
            newAngle = sSmoothedAngle + diff;
        }
        sSmoothedAngle = newAngle;
    }

    /***********************************************************************
     * 6. ROTATE ORIGINAL IMAGE USING sSmoothedAngle (GPU)
     ***********************************************************************/
    cv::Point2f center(input.cols / 2.0f, input.rows / 2.0f);

    // Positive angle => rotate CCW in standard OpenCV
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, sSmoothedAngle, 1.0);

    cv::cuda::GpuMat gpuMapX, gpuMapY, gpuRotated;
    cv::cuda::buildWarpAffineMaps(rotationMatrix, false, input.size(), gpuMapX, gpuMapY);
    cv::cuda::remap(gpuInput, gpuRotated, gpuMapX, gpuMapY, 
                    cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // Download final
    cv::Mat corrected;
    gpuRotated.download(corrected);
    return corrected;
}

} // namespace vs
