#ifndef AUTO_ZOOM_CROP_H
#define AUTO_ZOOM_CROP_H

#include <opencv2/opencv.hpp>

namespace vs {
    class AutoZoomCrop {
    public:
        /**
         * @brief Hide black corners introduced by roll correction by cropping + scaling.
         * @param corrected      The roll-corrected image (BGR or grayscale).
         * @param marginPercent  Extra margin to keep around the detected region (default: 5%).
         * @return               A cropped-and-scaled image, preserving aspect ratio.
         */
        static cv::Mat autoZoomCrop(const cv::Mat& corrected, double marginPercent = 0.05);
    };
}

#endif // AUTO_ZOOM_CROP_H
