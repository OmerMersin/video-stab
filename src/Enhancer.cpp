// video/Enhancer.cpp
#include "video/Enhancer.h"

// CPU modules
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
// GPU modules
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <cmath>
#include <vector>
#include <algorithm>  // For std::max
#include <opencv2/opencv.hpp>

namespace vs {

/* ----------------- CPU HELPER: White Balance ----------------- */
void Enhancer::whiteBalanceCPU(cv::Mat& bgr, float alpha) {
    if (bgr.channels() != 3) return;
    std::vector<cv::Mat> ch(3);
    cv::split(bgr, ch);
    double meanB = cv::mean(ch[0])[0];
    double meanG = cv::mean(ch[1])[0];
    double meanR = cv::mean(ch[2])[0];
    double gray = (meanB + meanG + meanR) / 3.0;
    double scaleB = gray / (meanB + 1e-6);
    double scaleG = gray / (meanG + 1e-6);
    double scaleR = gray / (meanR + 1e-6);
    scaleB = 1.0 + alpha * (scaleB - 1.0);
    scaleG = 1.0 + alpha * (scaleG - 1.0);
    scaleR = 1.0 + alpha * (scaleR - 1.0);
    ch[0] *= scaleB;
    ch[1] *= scaleG;
    ch[2] *= scaleR;
    cv::merge(ch, bgr);
}

/* ----------------- CPU HELPER: Vibrance ----------------- */
void Enhancer::vibranceCPU(cv::Mat& bgr, float alpha) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels(3);
    cv::split(hsv, channels);
    for (int r = 0; r < hsv.rows; r++) {
        uchar* sRow = channels[1].ptr<uchar>(r);
        for (int c = 0; c < hsv.cols; c++) {
            float sVal = static_cast<float>(sRow[c]);
            sVal += alpha * (255.f - sVal);
            sRow[c] = cv::saturate_cast<uchar>(sVal);
        }
    }
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
}

/* ----------------- CPU HELPER: CLAHE ----------------- */
void Enhancer::applyClaheCPU(cv::Mat& bgr, float clipLimit, int tileGridSize) {
    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> ch(3);
    cv::split(lab, ch);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tileGridSize, tileGridSize));
    clahe->apply(ch[0], ch[0]);
    cv::merge(ch, lab);
    cv::cvtColor(lab, bgr, cv::COLOR_Lab2BGR);
}

/* ----------------- GPU HELPER: White Balance ----------------- */
void Enhancer::whiteBalanceGPU(cv::cuda::GpuMat& frame, float alpha, cv::cuda::Stream& stream) {
    if (frame.channels() != 3) return;
    std::vector<cv::cuda::GpuMat> ch;
    cv::cuda::split(frame, ch, stream);
    stream.waitForCompletion(); // Ensure splits are done
    double sumB = cv::cuda::sum(ch[0])[0];
    double sumG = cv::cuda::sum(ch[1])[0];
    double sumR = cv::cuda::sum(ch[2])[0];
    double totalPixels = static_cast<double>(frame.cols * frame.rows);
    double meanB = sumB / totalPixels;
    double meanG = sumG / totalPixels;
    double meanR = sumR / totalPixels;
    double gray = (meanB + meanG + meanR) / 3.0;
    double scaleB = gray / (meanB + 1e-6);
    double scaleG = gray / (meanG + 1e-6);
    double scaleR = gray / (meanR + 1e-6);
    scaleB = 1.0 + alpha * (scaleB - 1.0);
    scaleG = 1.0 + alpha * (scaleG - 1.0);
    scaleR = 1.0 + alpha * (scaleR - 1.0);
    ch[0].convertTo(ch[0], -1, scaleB, 0, stream);
    ch[1].convertTo(ch[1], -1, scaleG, 0, stream);
    ch[2].convertTo(ch[2], -1, scaleR, 0, stream);
    cv::cuda::merge(ch, frame, stream);
}

/* ----------------- GPU HELPER: Vibrance ----------------- */
void Enhancer::vibranceGPU(cv::cuda::GpuMat& bgr, float alpha, cv::cuda::Stream& stream) {
    cv::cuda::GpuMat hsv;
    cv::cuda::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV, 0, stream);
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(hsv, channels, stream);
    stream.waitForCompletion();
    cv::Mat sCPU;
    channels[1].download(sCPU, stream);
    stream.waitForCompletion();
    for (int r = 0; r < sCPU.rows; r++) {
        uchar* rowPtr = sCPU.ptr<uchar>(r);
        for (int c = 0; c < sCPU.cols; c++) {
            float sVal = static_cast<float>(rowPtr[c]);
            sVal += alpha * (255.f - sVal);
            rowPtr[c] = cv::saturate_cast<uchar>(sVal);
        }
    }
    channels[1].upload(sCPU, stream);
    cv::cuda::merge(channels, hsv, stream);
    cv::cuda::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR, 0, stream);
}

/* ----------------- GPU HELPER: Unsharp Mask ----------------- */
void Enhancer::unsharpMaskGPU(cv::cuda::GpuMat& frame, float sharpness, float sigma, cv::cuda::Stream& stream) {
    if (sharpness <= 0.f) return;
    cv::cuda::GpuMat blurred;
    auto gauss = cv::cuda::createGaussianFilter(frame.type(), frame.type(), cv::Size(0,0), sigma);
    gauss->apply(frame, blurred, stream);
    cv::cuda::addWeighted(frame, 1.0 + sharpness, blurred, -sharpness, 0.0, frame, -1, stream);
}

/* ----------------- GPU HELPER: Denoise ----------------- */
void Enhancer::denoiseGPU(cv::cuda::GpuMat& frame, float strength, cv::cuda::Stream& stream) {
    if (strength <= 0.f) return;
    cv::cuda::GpuMat denoised;
    cv::cuda::fastNlMeansDenoisingColored(frame, denoised, strength, strength, 7, 21, stream);
    frame = denoised;
}

/* ----------------- MAIN enhanceImage FUNCTION ----------------- */
cv::Mat Enhancer::enhanceImage(const cv::Mat& input, const Parameters& params) {
    if (input.empty()) {
        return cv::Mat();
    }

    // ----- CPU PATH (if useCuda is false) -----
    if (!params.useCuda) {
        cv::Mat img = input.clone();

        if (params.enableWhiteBalance)
            whiteBalanceCPU(img, params.wbStrength);

        // Adjust brightness and contrast.
        img.convertTo(img, -1, params.contrast, params.brightness);

        if (params.enableClahe)
            applyClaheCPU(img, params.claheClipLimit, params.claheTileGridSize);

        if (params.enableVibrance)
            vibranceCPU(img, params.vibranceStrength);

        if (params.enableUnsharp && params.sharpness > 0.f) {
            cv::Mat blurred;
            cv::GaussianBlur(img, blurred, cv::Size(0,0), params.blurSigma);
            cv::addWeighted(img, 1.0 + params.sharpness, blurred, -params.sharpness, 0, img);
        }

        if (params.enableDenoise && params.denoiseStrength > 0.f) {
            cv::fastNlMeansDenoisingColored(img, img,
                                            params.denoiseStrength,
                                            params.denoiseStrength, 7, 21);
        }

        if (std::fabs(params.gamma - 1.f) > 1e-3) {
            // Precompute LUT for gamma correction.
            cv::Mat lut(1, 256, CV_8U);
            for (int i = 0; i < 256; ++i) {
                float norm = i / 255.f;
                float corrected = std::pow(norm, params.gamma);
                lut.at<uchar>(i) = cv::saturate_cast<uchar>(corrected * 255.f);
            }
            cv::LUT(img, lut, img);
        }
        return img;
    }

    // ----- GPU PATH -----
    cv::cuda::Stream stream;
    cv::cuda::GpuMat gpuFrame;
    gpuFrame.upload(input, stream);

    // Adjust brightness and contrast (asynchronously).
    gpuFrame.convertTo(gpuFrame, -1, params.contrast, params.brightness, stream);

    // Unsharp mask (GPU)
    if (params.enableUnsharp && params.sharpness > 0.f)
        unsharpMaskGPU(gpuFrame, params.sharpness, params.blurSigma, stream);

    // Denoise (GPU)
    if (params.enableDenoise && params.denoiseStrength > 0.f)
        denoiseGPU(gpuFrame, params.denoiseStrength, stream);

    // GPU white balance and vibrance
    if (params.enableWhiteBalance)
        whiteBalanceGPU(gpuFrame, params.wbStrength, stream);
    if (params.enableVibrance)
        vibranceGPU(gpuFrame, params.vibranceStrength, stream);

    // For operations not available on GPU (CLAHE and gamma correction),
    // download once, process on CPU, then re-upload.
    bool fallback = (params.enableClahe || std::fabs(params.gamma - 1.f) > 1e-3);
    if (fallback) {
        cv::Mat hostFrame;
        gpuFrame.download(hostFrame, stream);
        stream.waitForCompletion();

        if (params.enableClahe)
            applyClaheCPU(hostFrame, params.claheClipLimit, params.claheTileGridSize);

        if (std::fabs(params.gamma - 1.f) > 1e-3) {
            // Cache the LUT in static variables to avoid re-computation if gamma hasn’t changed.
            static double lastGamma = -1.0;
            static cv::Mat gammaLUT;
            if (std::fabs(params.gamma - lastGamma) > 1e-3) {
                gammaLUT = cv::Mat(1, 256, CV_8U);
                for (int i = 0; i < 256; ++i) {
                    float norm = i / 255.f;
                    float corrected = std::pow(norm, params.gamma);
                    gammaLUT.at<uchar>(i) = cv::saturate_cast<uchar>(corrected * 255.f);
                }
                lastGamma = params.gamma;
            }
            cv::LUT(hostFrame, gammaLUT, hostFrame);
        }
        gpuFrame.upload(hostFrame, stream);
    }

    cv::Mat result;
    gpuFrame.download(result, stream);
    stream.waitForCompletion();
    return result;
}

} // namespace vs
