#ifndef ENHANCER_H
#define ENHANCER_H

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace vs {

class Enhancer {
public:
    struct Parameters {
        // Basic
        float brightness   = 0.0f;  // add to pixel intensity
        float contrast     = 1.0f;  // multiply pixel intensity

        // White balance
        bool  enableWhiteBalance = false;
        float wbStrength         = 1.0f;  

        // Vibrance
        bool  enableVibrance     = false;
        float vibranceStrength   = 0.3f;

        // Unsharp
        bool  enableUnsharp      = false;
        float sharpness          = 0.0f;
        float blurSigma          = 1.0f;

        // CLAHE
        bool  enableClahe        = false;
        float claheClipLimit     = 2.0f;
        int   claheTileGridSize  = 8;

        // Denoise
        bool  enableDenoise      = false;
        float denoiseStrength    = 10.0f;

        // Gamma
        float gamma              = 1.0f;

        // CUDA
        bool  useCuda            = false;
    };

    static cv::Mat enhanceImage(const cv::Mat& input, const Parameters& params);

private:
    // CPU versions
    static void whiteBalanceCPU(cv::Mat& bgr, float alpha);
    static void vibranceCPU(cv::Mat& bgr, float alpha);
    static void applyClaheCPU(cv::Mat& bgr, float clipLimit, int tileGridSize);

    // GPU-based partial
    static void whiteBalanceGPU(cv::cuda::GpuMat& bgr, float alpha, cv::cuda::Stream& stream);
    static void vibranceGPU(cv::cuda::GpuMat& bgr, float alpha, cv::cuda::Stream& stream);
    
    // GPU-only steps
    static void unsharpMaskGPU(cv::cuda::GpuMat& frame, float sharpness, float sigma, cv::cuda::Stream& stream);
    static void denoiseGPU(cv::cuda::GpuMat& frame, float strength, cv::cuda::Stream& stream);
    static void gammaCorrectionGPU(cv::cuda::GpuMat& frame, float gammaVal, cv::cuda::Stream& stream);
};

} // namespace vs

#endif // ENHANCER_H
