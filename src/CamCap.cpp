#include "video/CamCap.h"
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace vs {
    static std::map<std::string, int> COLORSPACE_CODES = {
        {"BGR2GRAY", cv::COLOR_BGR2GRAY},
        {"BGR2HSV",  cv::COLOR_BGR2HSV},
        {"BGR2YUV",  cv::COLOR_BGR2YUV},
    };

    CamCap::CamCap(const Parameters& params) : params(params) {
        // Logging
        if(params.logging) {
            std::cout << "[CamCap] Initializing CamCap with source: " 
                    << params.source << std::endl;
        }

        // Determine if source is numeric (camera index) or not
        bool isNumeric = true;
        for(char c : params.source) {
            if(!isdigit(c)) {
                isNumeric = false;
                break;
            }
        }

        // Open the capture
        if(isNumeric && !params.source.empty()) {
            int camIndex = std::stoi(params.source);
            cap.open(camIndex, params.backend);
        } 
        else if (params.source.rfind("rtsp", 0) == 0) {  // Check if source starts with "rtsp"
            if (params.logging) {
                std::cout << "[CamCap] RTSP stream detected! Using GStreamer pipeline." << std::endl;
            }

            std::string gst_pipeline = "rtspsrc location=" + params.source + " latency=0 ! "
                                    "rtph265depay ! avdec_h265 ! videoconvert ! appsink";

            cap.open(gst_pipeline, cv::CAP_GSTREAMER);
        }
        else {
            // It might be a file path, RTSP URL, HTTP stream, etc.
            cap.open(params.source, params.backend);
        }

        if(!cap.isOpened()) {
            throw std::runtime_error("[CamCap] Failed to open source: " + params.source);
        }

        // If user specified a colorspace
        if(!params.colorspace.empty()) {
            auto it = COLORSPACE_CODES.find(params.colorspace);
            if(it != COLORSPACE_CODES.end()) {
                colorConversionCode = it->second;
                if(params.logging) {
                    std::cout << "[CamCap] Using colorspace: " 
                            << params.colorspace << std::endl;
                }
            } else {
                if(params.logging) {
                    std::cerr << "[CamCap] Warning: Invalid colorspace " 
                            << params.colorspace << " ignored." << std::endl;
                }
            }
        }

        // Retrieve framerate if available
        framerate = cap.get(cv::CAP_PROP_FPS);
        if(framerate < 1.0) {
            // fallback in case FPS is invalid
            framerate = 0.0; 
        }

        // Warm-up if requested
        if(params.timeDelay > 0) {
            if(params.logging) {
                std::cout << "[CamCap] Warming up for " 
                        << params.timeDelay << " seconds..." << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::seconds(params.timeDelay));
        }

        // Grab an initial frame to confirm validity
        if(!cap.read(currentFrame) || currentFrame.empty()) {
            throw std::runtime_error("[CamCap] Failed to read initial frame!");
        }

        // Optional: Apply color conversion once to confirm it works
        if(colorConversionCode != -1) {
    #ifdef HAVE_OPENCV_CUDACODEC
            cudaConvertColor(currentFrame);
    #else
            cv::cvtColor(currentFrame, currentFrame, colorConversionCode);
    #endif
        }

        // If threaded queue mode is on, push that first frame
        if(params.threadedQueueMode) {
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(currentFrame.clone());
        }
    }

    CamCap::~CamCap() {
        stop();  // Ensure resources are freed
    }

    void CamCap::start() {
        if(params.threadedQueueMode && !isRunning) {
            // Spawn capturing thread
            terminate = false;
            isRunning = true;
            captureThread = std::thread(&CamCap::updateLoop, this);

            if(params.logging) {
                std::cout << "[CamCap] Started capture thread." << std::endl;
            }
        }
        // If not threaded, we do nothing here. User calls read() which reads directly.
    }

    void CamCap::updateLoop() {
        while(!terminate) {
            cv::Mat frame;
            if(!cap.read(frame) || frame.empty()) {
                // No more frames or stream ended
                if(params.logging) {
                    std::cout << "[CamCap] No more frames, breaking thread loop." << std::endl;
                }
                break;
            }

            // If color conversion requested
            if(colorConversionCode != -1) {
    #ifdef HAVE_OPENCV_CUDACODEC
                cudaConvertColor(frame);
    #else
                cv::cvtColor(frame, frame, colorConversionCode);
    #endif
            }

            // Protect queue
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                // If queue is full, wait until there's space (similar to Python blocking)
                while(frameQueue.size() >= static_cast<size_t>(params.queueSize) && !terminate) {
                    // block until a frame is popped
                    queueCondition.wait(lock);
                }

                if(terminate) {
                    // If asked to terminate while waiting
                    break;
                }

                // Push the frame
                frameQueue.push(frame);
                // Notify a waiting consumer
                queueCondition.notify_one();
            }
        }

        // Indicate end by pushing an empty Mat
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(cv::Mat());
            queueCondition.notify_all();
        }

        if(params.logging) {
            std::cout << "[CamCap] Capture thread terminating." << std::endl;
        }
        isRunning = false;
    }

    cv::Mat CamCap::read() {
        if(params.threadedQueueMode) {
            // In threaded mode, pop from queue
            cv::Mat frame;
            std::unique_lock<std::mutex> lock(queueMutex);

            // If no-timeout scenario
            bool noTimeout = (params.threadTimeout <= 0);

            // We keep waiting until we get a frame or we see an empty marker
            auto startTime = std::chrono::steady_clock::now();

            while(frameQueue.empty() && !terminate) {
                if(noTimeout) {
                    queueCondition.wait(lock);
                } else {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - startTime).count();
                    if(elapsed > params.threadTimeout) {
                        // Timed out
                        if(params.logging) {
                            std::cerr << "[CamCap] Timed out waiting for frame in queue." << std::endl;
                        }
                        return cv::Mat(); // return empty
                    }
                    // Wait 1 ms and check again
                    queueCondition.wait_for(lock, std::chrono::milliseconds(1));
                }
            }

            if(!frameQueue.empty()) {
                frame = frameQueue.front();
                frameQueue.pop();
                // Notify producer that space is free
                queueCondition.notify_one();
            }
            return frame;

        } else {
            // Non-threaded: read directly from capture
            // (like python’s fallback if queue mode is disabled)
            if(!cap.isOpened()) {
                return cv::Mat();
            }

            cv::Mat frame;
            if(!cap.read(frame) || frame.empty()) {
                // No more frames
                return cv::Mat();
            }

            // Apply color conversion if needed
            if(colorConversionCode != -1) {
    #ifdef HAVE_OPENCV_CUDACODEC
                cudaConvertColor(frame);
    #else
                cv::cvtColor(frame, frame, colorConversionCode);
    #endif
            }
            currentFrame = frame.clone();
            return frame;
        }
    }

    void CamCap::stop() {
        if(params.threadedQueueMode) {
            if(isRunning) {
                if(params.logging) {
                    std::cout << "[CamCap] Stopping capture thread..." << std::endl;
                }
                terminate = true; // Signal thread
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    // Wake up if it's waiting
                    queueCondition.notify_all();
                }
                // Join thread
                if(captureThread.joinable()) {
                    captureThread.join();
                }
                isRunning = false;
            }
        }
        // Release capture
        if(cap.isOpened()) {
            cap.release();
        }

        // Clear queue
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            while(!frameQueue.empty()) {
                frameQueue.pop();
            }
        }
    }

    #ifdef HAVE_OPENCV_CUDACODEC
    void CamCap::cudaConvertColor(cv::Mat& frame) {
        // Upload to GPU
        cv::cuda::GpuMat gpuFrame, gpuOut;
        gpuFrame.upload(frame, cudaStream);

        // Convert color
        cv::cuda::cvtColor(gpuFrame, gpuOut, colorConversionCode, 0, cudaStream);

        // Download to CPU
        gpuOut.download(frame, cudaStream);

        // Ensure all operations complete before returning
        cudaStream.waitForCompletion();
    }
    #endif
}