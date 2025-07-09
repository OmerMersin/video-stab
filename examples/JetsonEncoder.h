#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

// Forward declarations for Jetson Multimedia API
class NvVideoEncoder;

/**
 * Hardware-accelerated H.264/H.265 encoder using Jetson Multimedia API
 * This provides much better performance than software encoding
 */
class JetsonEncoder {
public:
    struct Config {
        int width = 1920;
        int height = 1080;
        int fps = 30;
        int bitrate = 4000000;  // In bps
        bool useH265 = false;   // false = H.264, true = H.265
        std::string preset = "ultrafast";
        bool lowLatency = true;
    };

    JetsonEncoder();
    ~JetsonEncoder();

    // Initialize the encoder with given configuration
    bool initialize(const Config& config);

    // Encode a BGR frame and return encoded data
    // Returns empty vector on failure
    std::vector<uint8_t> encodeFrame(const cv::Mat& bgrFrame);

    // Get SPS/PPS data for stream initialization
    std::vector<uint8_t> getExtraData() const;

    // Check if encoder is ready
    bool isReady() const { return initialized; }

    // Get current configuration
    const Config& getConfig() const { return config; }

private:
    bool initialized;
    Config config;
    
    // Jetson Multimedia API objects
    std::unique_ptr<NvVideoEncoder> encoder;
    
    // Helper methods
    bool setupEncoder();
    bool convertBGRToNV12(const cv::Mat& bgr, cv::Mat& nv12);
    void cleanup();
    
    // Buffers
    cv::Mat nv12Frame;
    std::vector<uint8_t> encodedBuffer;
    
    // Internal buffers
    cv::Mat nv12Frame;
    std::vector<uint8_t> encodedBuffer;
    
    // Helper methods
    bool setupEncoder();
    void cleanup();
    bool convertBGRToNV12(const cv::Mat& bgr, cv::Mat& nv12);
};
