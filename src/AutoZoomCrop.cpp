#include "video/AutoZoomCrop.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <algorithm>

// Keep the same helper functions:
bool checkInteriorExterior(const cv::Mat &mask, const cv::Rect &interiorBB, int &top, int &bottom, int &left, int &right)
{
    bool returnVal = true;
    cv::Mat sub = mask(interiorBB);

    unsigned int x=0;
    unsigned int y=0;

    unsigned int cTop=0;    // top row
    unsigned int cBottom=0; // bottom row
    unsigned int cLeft=0;   // left column
    unsigned int cRight=0;  // right column

    for(y=0, x=0 ; x<sub.cols; ++x)
    {
        if(sub.at<unsigned char>(y,x) == 0)
        {
            returnVal = false;
            ++cTop;
        }
    }
    for(y=sub.rows-1, x=0; x<sub.cols; ++x)
    {
        if(sub.at<unsigned char>(y,x) == 0)
        {
            returnVal = false;
            ++cBottom;
        }
    }
    for(y=0, x=0 ; y<sub.rows; ++y)
    {
        if(sub.at<unsigned char>(y,x) == 0)
        {
            returnVal = false;
            ++cLeft;
        }
    }
    for(x=sub.cols-1, y=0; y<sub.rows; ++y)
    {
        if(sub.at<unsigned char>(y,x) == 0)
        {
            returnVal = false;
            ++cRight;
        }
    }

    // Decide which border to shrink
    if(cTop > cBottom)
    {
        if(cTop > cLeft)
            if(cTop > cRight)
                top = 1;
    }
    else
        if(cBottom > cLeft)
            if(cBottom > cRight)
                bottom = 1;

    if(cLeft >= cRight)
    {
        if(cLeft >= cBottom)
            if(cLeft >= cTop)
                left = 1;
    }
    else
        if(cRight >= cTop)
            if(cRight >= cBottom)
                right = 1;

    return returnVal;
}

bool sortX(cv::Point a, cv::Point b)
{
    bool ret = false;
    if(a.x == a.x)
        if(b.x == b.x)
            ret = a.x < b.x;
    return ret;
}

bool sortY(cv::Point a, cv::Point b)
{
    bool ret = false;
    if(a.y == a.y)
        if(b.y == b.y)
            ret = a.y < b.y;
    return ret;
}

namespace vs {

cv::Mat AutoZoomCrop::autoZoomCrop(const cv::Mat &corrected, double /*marginPercent*/)
{
    // Basic sanity check
    if (corrected.empty()) {
        return corrected.clone();
    }

    // --- 0. Upload to GPU
    cv::cuda::GpuMat d_corrected;
    d_corrected.upload(corrected);

    // --- 1. Convert to grayscale if needed (on GPU)
    cv::cuda::GpuMat d_gray;
    if (corrected.channels() == 3) {
        cv::cuda::cvtColor(d_corrected, d_gray, cv::COLOR_BGR2GRAY);
    } else {
        d_gray = d_corrected.clone();
    }
    // --- 2. Threshold to isolate black corners (on GPU)
    cv::cuda::GpuMat d_mask;
    cv::cuda::threshold(d_gray, d_mask, /*thresh=*/1, /*maxval=*/255, cv::THRESH_BINARY_INV);

    // --- 3. Morphological close on d_mask (on GPU)
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));

cv::Ptr<cv::cuda::Filter> morphCloseMask = cv::cuda::createMorphologyFilter(
    cv::MORPH_CLOSE, d_mask.type(), kernel);  // ✅ Use CPU `cv::Mat`, not `GpuMat`
morphCloseMask->apply(d_mask, d_mask);


    // --- 4. Get contentMask by thresholding for non-black content (on GPU)
    cv::cuda::GpuMat d_contentMask;
    cv::cuda::threshold(d_gray, d_contentMask, /*thresh=*/1, /*maxval=*/255, cv::THRESH_BINARY);

    // Morphological close on d_contentMask (on GPU)
    cv::Ptr<cv::cuda::Filter> morphCloseContent = cv::cuda::createMorphologyFilter(
        cv::MORPH_CLOSE, d_contentMask.type(), kernel);
    morphCloseContent->apply(d_contentMask, d_contentMask);

    // --- 5. Download contentMask for CPU-based findContours
    cv::Mat contentMask;
    d_contentMask.download(contentMask);

    // Find contours on CPU
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(contentMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if(contours.empty()) {
        // No valid contours found, just return original
        return corrected.clone();
    }

    // Find largest contour by size
    unsigned int maxSize = 0;
    unsigned int id = 0;
    for(unsigned int i=0; i<contours.size(); ++i)
    {
        if(contours[i].size() > maxSize)
        {
            maxSize = (unsigned int)contours[i].size();
            id = i;
        }
    }

    // Create a contour mask for that largest contour
    cv::Mat contourMask = cv::Mat::zeros(corrected.size(), CV_8UC1);
    cv::drawContours(contourMask, contours, (int)id, cv::Scalar(255), -1);

    // Sort contour in x and y directions
    std::vector<cv::Point> cSortedX = contours[id];
    std::sort(cSortedX.begin(), cSortedX.end(), sortX);

    std::vector<cv::Point> cSortedY = contours[id];
    std::sort(cSortedY.begin(), cSortedY.end(), sortY);

    unsigned int minXId = 0;
    unsigned int maxXId = (unsigned int)(cSortedX.size() - 1);

    unsigned int minYId = 0;
    unsigned int maxYId = (unsigned int)(cSortedY.size() - 1);

    cv::Rect interiorBB;

// Get the original aspect ratio (width/height)
double originalAR = static_cast<double>(corrected.cols) / corrected.rows;

// Compute bounding box inside contour
while ((minXId < maxXId) && (minYId < maxYId)) {
    cv::Point minPt(cSortedX[minXId].x, cSortedY[minYId].y);
    cv::Point maxPt(cSortedX[maxXId].x, cSortedY[maxYId].y);

    interiorBB = cv::Rect(minPt.x, minPt.y, maxPt.x - minPt.x, maxPt.y - minPt.y);

    int ocTop = 0, ocBottom = 0, ocLeft = 0, ocRight = 0;
    bool finished = checkInteriorExterior(contourMask, interiorBB, ocTop, ocBottom, ocLeft, ocRight);
    if (finished) {
        break;
    }
    if (ocLeft) ++minXId;
    if (ocRight) --maxXId;
    if (ocTop) ++minYId;
    if (ocBottom) --maxYId;
}

// Calculate new width to maintain original aspect ratio
int newWidth = static_cast<int>(interiorBB.height * originalAR);

// Center the new width within the bounding box
int centerX = interiorBB.x + interiorBB.width / 2;
interiorBB.width = newWidth;
interiorBB.x = centerX - (newWidth / 2);

// Ensure it's within the image bounds
if (interiorBB.x < 0) interiorBB.x = 0;
if (interiorBB.x + interiorBB.width > corrected.cols)
    interiorBB.x = corrected.cols - interiorBB.width;

// Final check to ensure it's still inside the image
interiorBB &= cv::Rect(0, 0, corrected.cols, corrected.rows);

std::cout << "Final crop: " << interiorBB.width << "x" << interiorBB.height << std::endl;

    // Ensure bounding box is valid
    cv::Rect validRect = interiorBB & cv::Rect(0, 0, corrected.cols, corrected.rows);

    // Crop on GPU
    cv::cuda::GpuMat d_cropped = d_corrected(validRect);

    // --- GPU-based "resize" using warpAffine (since cv::cuda::resize doesn't exist in official OpenCV):
    cv::cuda::GpuMat d_resized;
    // Ensure d_cropped is not empty before processing
if (d_cropped.empty()) {
    cv::Mat fallback;
    d_corrected.download(fallback);
    return fallback;  // Return original image if cropping failed
}

// Validate scaling factors
if (d_cropped.cols == 0 || d_cropped.rows == 0) {
    cv::Mat fallback;
    d_corrected.download(fallback);
    return fallback;  // Return original image
}

double scaleX = 640.0 / d_cropped.cols;
double scaleY = 360.0 / d_cropped.rows;

// Prevent division by zero or invalid scaling
if (scaleX <= 0 || scaleY <= 0 || std::isinf(scaleX) || std::isinf(scaleY)) {
    cv::Mat fallback;
    d_corrected.download(fallback);
    return fallback;
}

// Create 2x3 affine transform for scaling
cv::Mat h_M = (cv::Mat_<float>(2,3) << (float)scaleX, 0.0f, 0.0f,
                                       0.0f, (float)scaleY, 0.0f);

// Ensure destination size is valid
cv::Size outputSize(640, 360);
if (outputSize.width <= 0 || outputSize.height <= 0) {
    cv::Mat fallback;
    d_corrected.download(fallback);
    return fallback;
}

// Perform warpAffine with error checking
try {
    cv::cuda::warpAffine(d_cropped, d_resized, h_M, outputSize, cv::INTER_LINEAR);
} catch (const cv::Exception &e) {
    std::cerr << "warpAffine failed: " << e.what() << std::endl;
    cv::Mat fallback;
    d_corrected.download(fallback);
    return fallback;
}

// Download back to CPU
cv::Mat resizedImage;
d_resized.download(resizedImage);

return resizedImage;
}

} // namespace vs
