#include "JetsonEncoder.h"
#include <iostream>
#include <cstring>

// Jetson Multimedia API includes
#include "NvVideoEncoder.h"
#include "NvUtils.h"
#include "NvJpegEncoder.h"

JetsonEncoder::JetsonEncoder() : initialized(false) {
}

JetsonEncoder::~JetsonEncoder() {
    cleanup();
}

bool JetsonEncoder::initialize(const Config& cfg) {
    config = cfg;
    
    try {
        // Create encoder instance
        encoder = std::make_unique<NvVideoEncoder>("enc0");
        if (!encoder) {
            std::cerr << "Failed to create NvVideoEncoder" << std::endl;
            return false;
        }

        // Set encoder format
        uint32_t pixfmt = config.useH265 ? V4L2_PIX_FMT_H265 : V4L2_PIX_FMT_H264;
        if (encoder->setCapturePlaneFormat(pixfmt, config.width, config.height, 2 * 1024 * 1024) < 0) {
            std::cerr << "Failed to set capture plane format" << std::endl;
            return false;
        }

        // Set output plane format (NV12)
        if (encoder->setOutputPlaneFormat(V4L2_PIX_FMT_NV12M, config.width, config.height) < 0) {
            std::cerr << "Failed to set output plane format" << std::endl;
            return false;
        }

        // Set encoding parameters
        if (encoder->setBitrate(config.bitrate) < 0) {
            std::cerr << "Failed to set bitrate" << std::endl;
            return false;
        }

        if (encoder->setFrameRate(config.fps, 1) < 0) {
            std::cerr << "Failed to set frame rate" << std::endl;
            return false;
        }

        // Set low latency parameters
        if (config.lowLatency) {
            encoder->setIDRInterval(15);  // I-frame every 15 frames
            encoder->setIFrameInterval(15);
            
            // Disable B-frames for low latency
            encoder->setNumBFrames(0);
            
            // Set rate control mode to CBR for consistent latency
            encoder->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_CBR);
            
            // Set profile for low latency
            if (config.useH265) {
                encoder->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN);
                encoder->setLevel(V4L2_MPEG_VIDEO_H265_LEVEL_4_0);
            } else {
                encoder->setProfile(V4L2_MPEG_VIDEO_H264_PROFILE_HIGH);
                encoder->setLevel(V4L2_MPEG_VIDEO_H264_LEVEL_4_0);
            }
        }

        // Setup encoder
        if (!setupEncoder()) {
            std::cerr << "Failed to setup encoder" << std::endl;
            return false;
        }

        // Prepare conversion buffer
        nv12Frame = cv::Mat::zeros(config.height * 3 / 2, config.width, CV_8UC1);
        
        initialized = true;
        std::cout << "Jetson " << (config.useH265 ? "H.265" : "H.264") 
                  << " encoder initialized: " << config.width << "x" << config.height 
                  << " @ " << config.fps << " fps, " << config.bitrate << " bps" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in JetsonEncoder::initialize: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

bool JetsonEncoder::setupEncoder() {
    try {
        // Setup capture plane (encoded output)
        if (encoder->setupCapturePlane(V4L2_MEMORY_MMAP, 10, false) < 0) {
            std::cerr << "Failed to setup capture plane" << std::endl;
            return false;
        }

        // Setup output plane (raw input)
        if (encoder->setupOutputPlane(V4L2_MEMORY_MMAP, 10, true, false) < 0) {
            std::cerr << "Failed to setup output plane" << std::endl;
            return false;
        }

        // Start streaming
        if (encoder->startStream(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) < 0) {
            std::cerr << "Failed to start output stream" << std::endl;
            return false;
        }

        if (encoder->startStream(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) < 0) {
            std::cerr << "Failed to start capture stream" << std::endl;
            return false;
        }

        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in setupEncoder: " << e.what() << std::endl;
        return false;
    }
}

std::vector<uint8_t> JetsonEncoder::encodeFrame(const cv::Mat& bgrFrame) {
    std::vector<uint8_t> result;
    
    if (!initialized || bgrFrame.empty()) {
        return result;
    }

    try {
        // Convert BGR to NV12
        cv::Mat nv12;
        if (!convertBGRToNV12(bgrFrame, nv12)) {
            std::cerr << "Failed to convert BGR to NV12" << std::endl;
            return result;
        }

        // Get input buffer
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer* nvBuffer = nullptr;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.m.planes = planes;

        if (encoder->getOutputPlaneBuffer(&v4l2_buf, &nvBuffer) < 0) {
            std::cerr << "Failed to get output buffer" << std::endl;
            return result;
        }

        // Copy NV12 data to buffer
        if (nvBuffer->n_planes == 2) {
            // Y plane
            memcpy(nvBuffer->planes[0].data, nv12.data, config.width * config.height);
            // UV plane
            memcpy(nvBuffer->planes[1].data, nv12.data + config.width * config.height, 
                   config.width * config.height / 2);
        }

        // Queue buffer for encoding
        v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
        if (encoder->queueOutputPlaneBuffer(&v4l2_buf, -1) < 0) {
            std::cerr << "Failed to queue output buffer" << std::endl;
            return result;
        }

        // Get encoded output
        struct v4l2_buffer cap_v4l2_buf;
        NvBuffer* cap_nvBuffer = nullptr;
        
        memset(&cap_v4l2_buf, 0, sizeof(cap_v4l2_buf));
        
        if (encoder->dqCapturePlaneBuffer(&cap_v4l2_buf, &cap_nvBuffer, nullptr, -1) == 0) {
            // Copy encoded data
            if (cap_nvBuffer && cap_nvBuffer->planes[0].bytesused > 0) {
                result.resize(cap_nvBuffer->planes[0].bytesused);
                memcpy(result.data(), cap_nvBuffer->planes[0].data, cap_nvBuffer->planes[0].bytesused);
            }
            
            // Queue buffer back
            encoder->queueCapturePlaneBuffer(&cap_v4l2_buf, -1);
        }

        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in encodeFrame: " << e.what() << std::endl;
        return result;
    }
}

bool JetsonEncoder::convertBGRToNV12(const cv::Mat& bgr, cv::Mat& nv12) {
    if (bgr.empty() || bgr.cols != config.width || bgr.rows != config.height) {
        return false;
    }

    try {
        // Resize nv12 buffer if needed
        int nv12_size = config.width * config.height * 3 / 2;
        if (nv12.total() != nv12_size) {
            nv12 = cv::Mat::zeros(config.height * 3 / 2, config.width, CV_8UC1);
        }

        // Convert BGR to YUV420 (NV12)
        cv::Mat yuv420;
        cv::cvtColor(bgr, yuv420, cv::COLOR_BGR2YUV_I420);
        
        // Rearrange to NV12 format (Y plane + interleaved UV)
        uint8_t* nv12_data = nv12.data;
        uint8_t* yuv_data = yuv420.data;
        
        int y_size = config.width * config.height;
        int uv_size = y_size / 4;
        
        // Copy Y plane
        memcpy(nv12_data, yuv_data, y_size);
        
        // Interleave U and V planes
        uint8_t* u_plane = yuv_data + y_size;
        uint8_t* v_plane = yuv_data + y_size + uv_size;
        uint8_t* uv_plane = nv12_data + y_size;
        
        for (int i = 0; i < uv_size; i++) {
            uv_plane[i * 2] = u_plane[i];
            uv_plane[i * 2 + 1] = v_plane[i];
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in convertBGRToNV12: " << e.what() << std::endl;
        return false;
    }
}

std::vector<uint8_t> JetsonEncoder::getExtraData() const {
    std::vector<uint8_t> extraData;
    
    if (!initialized || !encoder) {
        return extraData;
    }
    
    // This would contain SPS/PPS for H.264 or VPS/SPS/PPS for H.265
    // Implementation depends on specific requirements
    // For now, return empty - the encoder will include headers in stream
    
    return extraData;
}

void JetsonEncoder::cleanup() {
    if (encoder) {
        try {
            encoder->stopStream(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
            encoder->stopStream(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
            encoder.reset();
        } catch (...) {
            // Ignore cleanup errors
        }
    }
    
    initialized = false;
    nv12Frame.release();
    encodedBuffer.clear();
}
