#include "video/AutoZoomCrop.h"
#include <opencv2/opencv.hpp>
#include <algorithm>

    bool checkInteriorExterior(const cv::Mat&mask, const cv::Rect&interiorBB, int&top, int&bottom, int&left, int&right)
{
// return true if the rectangle is fine as it is!
bool returnVal = true;

cv::Mat sub = mask(interiorBB);

unsigned int x=0;
unsigned int y=0;

// count how many exterior pixels are at the
unsigned int cTop=0; // top row
unsigned int cBottom=0; // bottom row
unsigned int cLeft=0; // left column
unsigned int cRight=0; // right column
// and choose that side for reduction where mose exterior pixels occured (that's the heuristic)

for(y=0, x=0 ; x<sub.cols; ++x)
{
    // if there is an exterior part in the interior we have to move the top side of the rect a bit to the bottom
    if(sub.at<unsigned char>(y,x) == 0)
    {
        returnVal = false;
        ++cTop;
    }
}

for(y=sub.rows-1, x=0; x<sub.cols; ++x)
{
    // if there is an exterior part in the interior we have to move the bottom side of the rect a bit to the top
    if(sub.at<unsigned char>(y,x) == 0)
    {
        returnVal = false;
        ++cBottom;
    }
}

for(y=0, x=0 ; y<sub.rows; ++y)
{
    // if there is an exterior part in the interior
    if(sub.at<unsigned char>(y,x) == 0)
    {
        returnVal = false;
        ++cLeft;
    }
}

for(x=sub.cols-1, y=0; y<sub.rows; ++y)
{
    // if there is an exterior part in the interior
    if(sub.at<unsigned char>(y,x) == 0)
    {
        returnVal = false;
        ++cRight;
    }
}

// that part is ugly and maybe not correct, didn't check whether all possible combinations are handled. Check that one please. The idea is to set `top = 1` iff it's better to reduce the rect at the top than anywhere else.
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
        if(b.x==b.x)
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
namespace vs{
    /**
     * @brief Hide black corners introduced by roll correction by cropping + scaling.
     * @param corrected      The roll-corrected image (BGR or grayscale).
     * @param marginPercent  Extra margin to keep around the detected region (0.05 = 5%).
     * @return               A cropped-and-scaled image, preserving aspect ratio.
     *
     * This function:
     *   1. Converts to grayscale if necessary.
     *   2. Thresholds to find black areas.
     *   3. Optionally does a morphological “close” to remove noise.
     *   4. Finds bounding rectangle around non-black region.
     *   5. Crops, then scales up to fit the original size, and centers (padded).
     */
    cv::Mat AutoZoomCrop::autoZoomCrop(const cv::Mat& corrected, double marginPercent)
    {
    // Basic sanity check
    if (corrected.empty()) {
        return corrected.clone();
    }

    // Convert to grayscale if needed
    cv::Mat gray;
    if (corrected.channels() == 3) {
        cv::cvtColor(corrected, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = corrected.clone();
    }

    // --- 1. Threshold to isolate black corners ---
    //    Here, we say anything “very dark” is black. Adjust threshold as needed.
    //    E.g., < 20 => consider it black. 
    cv::Mat mask;
    cv::threshold(gray, mask, /*thresh=*/1, /*maxval=*/255, cv::THRESH_BINARY_INV);

    // Now ‘mask’ has 255 where it's black/dark corners, 0 where it's bright content.

    // --- 2. (Optional) Morphological close to remove small holes or noise ---
    //    This step merges adjacent black regions so boundingRect is more robust.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // --- 3. Find bounding rectangle around the “non-black” content
    //        We want to invert so that we find the boundingRect of the actual content,
    //        not the black corners. An easy approach is to findContours on the inverse.
    cv::Mat contentMask;
    cv::threshold(gray, contentMask, /*thresh=*/1, /*maxval=*/255, cv::THRESH_BINARY); // now content=255, black=0

    // Optionally close on contentMask as well, to unify content regions:
    cv::morphologyEx(contentMask, contentMask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(contentMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
std::vector<cv::Vec4i> hierarchy;
std::cout << "found contours: " << contours.size() << std::endl;
cv::Mat contourImage = cv::Mat::zeros( corrected.size(), CV_8UC3 );;
unsigned int maxSize = 0;
unsigned int id = 0;
for(unsigned int i=0; i<contours.size(); ++i)
{
    if(contours.at(i).size() > maxSize)
    {
        maxSize = contours.at(i).size();
        id = i;
    }
}
std::cout << "chosen id: " << id << std::endl;
std::cout << "max size: " << maxSize << std::endl;
/// Draw filled contour to obtain a mask with interior parts
cv::Mat contourMask = cv::Mat::zeros( corrected.size(), CV_8UC1 );
cv::drawContours( contourMask, contours, id, cv::Scalar(255), -1, 8, hierarchy, 0, cv::Point() );
// cv::imshow("contour mask", contourMask);

// sort contour in x/y directions to easily find min/max and next
std::vector<cv::Point> cSortedX = contours.at(id);
std::sort(cSortedX.begin(), cSortedX.end(), sortX);

std::vector<cv::Point> cSortedY = contours.at(id);
std::sort(cSortedY.begin(), cSortedY.end(), sortY);


unsigned int minXId = 0;
unsigned int maxXId = cSortedX.size()-1;

unsigned int minYId = 0;
unsigned int maxYId = cSortedY.size()-1;

cv::Rect interiorBB;

while( (minXId<maxXId)&&(minYId<maxYId) )
{
    cv::Point min(cSortedX[minXId].x, cSortedY[minYId].y);
    cv::Point max(cSortedX[maxXId].x, cSortedY[maxYId].y);

    interiorBB = cv::Rect(min.x,min.y, max.x-min.x, max.y-min.y);

// out-codes: if one of them is set, the rectangle size has to be reduced at that border
    int ocTop = 0;
    int ocBottom = 0;
    int ocLeft = 0;
    int ocRight = 0;

    bool finished = checkInteriorExterior(contourMask, interiorBB, ocTop, ocBottom,ocLeft, ocRight);
    if(finished)
    {
        break;
    }

// reduce rectangle at border if necessary
    if(ocLeft)++minXId;
    if(ocRight) --maxXId;

    if(ocTop) ++minYId;
    if(ocBottom)--maxYId;


}
// cv::Mat mask2 = cv::Mat::zeros(corrected.rows, corrected.cols, CV_8UC1);
// cv::rectangle(mask2,interiorBB, cv::Scalar(255),-1);

// cv::Mat maskedImage;
// corrected.copyTo(maskedImage);
// for(unsigned int y=0; y<maskedImage.rows; ++y)
//     for(unsigned int x=0; x<maskedImage.cols; ++x)
//     {
//         maskedImage.at<cv::Vec3b>(y,x)[2] = 255;
//     }
// corrected.copyTo(maskedImage,mask2);
// cv::imshow("masked image", maskedImage);
// Ensure the cropped region maintains the original aspect ratio
cv::Mat mask2 = cv::Mat::zeros(corrected.rows, corrected.cols, CV_8UC1);
cv::rectangle(mask2, interiorBB, cv::Scalar(255), -1);  // Create a mask for the detected region

cv::Mat maskedImage;
corrected.copyTo(maskedImage, mask2); // Copy only the masked region

// Crop the region of interest (ROI)
cv::Mat croppedImage = corrected(interiorBB).clone();  // Ensure it's copied

// Resize to fit the display window
cv::Mat resizedImage;
cv::resize(croppedImage, resizedImage, cv::Size(640, 360), 0, 0, cv::INTER_LINEAR);  

cv::imshow("Cropped & Scaled", resizedImage);

float originalAspect = static_cast<float>(corrected.cols) / corrected.rows;
float newAspect = static_cast<float>(interiorBB.width) / interiorBB.height;

if (newAspect > originalAspect) {
    // New region is wider than the original aspect ratio, so increase height
    int newHeight = static_cast<int>(interiorBB.width / originalAspect);
    int extra = (newHeight - interiorBB.height) / 2;
    // interiorBB.y = std::max(0, interiorBB.y - extra);
    interiorBB.height = std::min(corrected.rows - interiorBB.y, newHeight);
} else {
    // New region is taller than the original aspect ratio, so increase width
    int newWidth = static_cast<int>(interiorBB.height * originalAspect);
    int extra = (newWidth - interiorBB.width) / 2;
    interiorBB.x = std::max(0, interiorBB.x - extra);
    interiorBB.width = std::min(corrected.cols - interiorBB.x, newWidth);
}
// interiorBB.y = 100;
// interiorBB.height = 200;
// interiorBB.x = 100;
// interiorBB.width = 200;

cv::Mat zoomed = corrected(interiorBB);
return zoomed;

    // If no contours found, just return the original corrected
    if (contours.empty()) {
        return corrected.clone();
    }

    // Compute the union of all contour bounding rects
    cv::Rect contentRect = cv::boundingRect(contours[0]);
    for (size_t i = 1; i < contours.size(); i++) {
        contentRect = contentRect | cv::boundingRect(contours[i]);
    }

    // --- 4. Expand bounding rectangle by some marginPercent (e.g., 5%) ---
    int marginX = static_cast<int>(contentRect.width  * marginPercent);
    int marginY = static_cast<int>(contentRect.height * marginPercent);

    contentRect.x      = std::max(0, contentRect.x - marginX);
    contentRect.y      = std::max(0, contentRect.y - marginY);
    contentRect.width  = std::min(corrected.cols - contentRect.x, contentRect.width  + 2*marginX);
    contentRect.height = std::min(corrected.rows - contentRect.y, contentRect.height + 2*marginY);

    // Crop
    cv::Mat cropped = corrected(contentRect).clone();

    // --- 5. Scale up so that cropped region fits the original size as much as possible ---
    double scale = std::min(
        static_cast<double>(corrected.cols) / cropped.cols,
        static_cast<double>(corrected.rows) / cropped.rows
    );
    // If you *don’t* want to enlarge beyond 100%, clamp scale to <= 1.0:
    // scale = std::min(scale, 1.0);

    cv::Size newSize(
        static_cast<int>(cropped.cols * scale),
        static_cast<int>(cropped.rows * scale)
    );
    cv::Mat scaled;
    cv::resize(cropped, scaled, newSize, 0, 0, cv::INTER_LINEAR);

    // --- 6. Center the scaled image in a canvas that matches the original dimensions ---
    cv::Mat finalResult = cv::Mat::zeros(corrected.size(), corrected.type());

    int offsetX = (corrected.cols - scaled.cols) / 2;
    int offsetY = (corrected.rows - scaled.rows) / 2;
    scaled.copyTo(finalResult(cv::Rect(offsetX, offsetY, scaled.cols, scaled.rows)));
std::cout << "YESS!!!" << "\n";
    return contentMask;
    }
}