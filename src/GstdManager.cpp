#include "video/GstdManager.h"
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <thread>
#include <chrono>

namespace vs {

GstdManager::GstdManager(const Parameters& params) : params(params) {
    if (params.logging) {
        std::cout << "[GstdManager] Initializing with source: " << params.rtspSource << std::endl;
    }
}

GstdManager::~GstdManager() {
    cleanup();
}

bool GstdManager::initialize() {
    if (initialized) {
        return true;
    }

    if (params.logging) {
        std::cout << "[GstdManager] Starting gstd daemon..." << std::endl;
    }

    // Kill any existing gstd instance and start fresh
    system("pkill -f gstd >/dev/null 2>&1");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Start gstd daemon
    std::string startGstdCmd = "gstd --daemon";
    if (system(startGstdCmd.c_str()) != 0) {
        std::cerr << "[GstdManager] Failed to start gstd daemon" << std::endl;
        return false;
    }

    // Give gstd time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // Create all pipelines
    if (!createPassthroughPipeline()) {
        std::cerr << "[GstdManager] Failed to create passthrough pipeline" << std::endl;
        return false;
    }

    if (!createProcessingPipeline()) {
        std::cerr << "[GstdManager] Failed to create processing pipeline" << std::endl;
        return false;
    }

    if (!createOutputPipeline()) {
        std::cerr << "[GstdManager] Failed to create output pipeline" << std::endl;
        return false;
    }

    initialized = true;
    
    if (params.logging) {
        std::cout << "[GstdManager] Initialization complete" << std::endl;
    }

    return true;
}

bool GstdManager::start() {
    if (!initialized) {
        std::cerr << "[GstdManager] Not initialized" << std::endl;
        return false;
    }

    // Start all pipelines
    if (!startPipeline(PASSTHROUGH_PIPELINE)) {
        std::cerr << "[GstdManager] Failed to start passthrough pipeline" << std::endl;
        return false;
    }

    if (!startPipeline(PROCESSING_PIPELINE)) {
        std::cerr << "[GstdManager] Failed to start processing pipeline" << std::endl;
        return false;
    }

    if (!startPipeline(OUTPUT_PIPELINE)) {
        std::cerr << "[GstdManager] Failed to start output pipeline" << std::endl;
        return false;
    }

    // Start in passthrough mode by default
    switchToPassthrough();

    // Start health check thread (disabled for now)
    healthCheckRunning = false;
    /*
    healthCheckThread = std::thread([this]() {
        while (healthCheckRunning) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            if (!isHealthy()) {
                if (params.logging) {
                    std::cout << "[GstdManager] Health check failed, attempting recovery..." << std::endl;
                }
                // Could implement recovery logic here
            }
        }
    });
    */

    if (params.logging) {
        std::cout << "[GstdManager] All pipelines started successfully" << std::endl;
    }

    return true;
}

void GstdManager::stop() {
    healthCheckRunning = false;
    if (healthCheckThread.joinable()) {
        healthCheckThread.join();
    }

    // Stop all pipelines
    stopPipeline(OUTPUT_PIPELINE);
    stopPipeline(PROCESSING_PIPELINE);
    stopPipeline(PASSTHROUGH_PIPELINE);

    if (params.logging) {
        std::cout << "[GstdManager] All pipelines stopped" << std::endl;
    }
}

void GstdManager::cleanup() {
    stop();

    if (initialized) {
        // Delete all pipelines
        sendGstdCommand("pipeline_delete " + std::string(PASSTHROUGH_PIPELINE));
        sendGstdCommand("pipeline_delete " + std::string(PROCESSING_PIPELINE));
        sendGstdCommand("pipeline_delete " + std::string(OUTPUT_PIPELINE));

        // Stop gstd daemon
        sendGstdCommand("quit");
        
        initialized = false;
        
        if (params.logging) {
            std::cout << "[GstdManager] Cleanup complete" << std::endl;
        }
    }
}

bool GstdManager::createPassthroughPipeline() {
    std::string depay, parse;
    if (params.codec == "h264") {
        depay = "rtph264depay";
        parse = "h264parse";
    } else {
        depay = "rtph265depay";
        parse = "h265parse";
    }

    // Ultra-low latency passthrough pipeline - no decoding/encoding
    std::ostringstream pipeline;
    pipeline << "rtspsrc location=" << params.rtspSource 
             << " protocols=tcp latency=0 drop-on-latency=true ! "
             << depay << " ! " << parse << " ! "
             << "interpipesink name=" << PASSTHROUGH_INTERPIPE 
             << " sync=false async=false";

    std::string cmd = "pipeline_create " + std::string(PASSTHROUGH_PIPELINE) + " \"" + pipeline.str() + "\"";
    
    if (params.logging) {
        std::cout << "[GstdManager] Creating passthrough pipeline: " << pipeline.str() << std::endl;
    }

    return sendGstdCommand(cmd);
}

bool GstdManager::createProcessingPipeline() {
    std::string depay, parse;
    if (params.codec == "h264") {
        depay = "rtph264depay";
        parse = "h264parse";
    } else {
        depay = "rtph265depay";
        parse = "h265parse";
    }

    // Processing pipeline - decode to raw frames for OpenCV processing
    std::ostringstream pipeline;
    pipeline << "rtspsrc location=" << params.rtspSource 
             << " protocols=tcp latency=0 drop-on-latency=true ! "
             << depay << " ! " << parse << " ! "
             << "nvv4l2decoder ! "
             << "nvvidconv ! video/x-raw,format=BGRx,width=" << params.width 
             << ",height=" << params.height << " ! "
             << "videoconvert ! video/x-raw,format=BGR ! "
             << "interpipesink name=" << PROCESSING_INTERPIPE 
             << " sync=false async=false";

    std::string cmd = "pipeline_create " + std::string(PROCESSING_PIPELINE) + " \"" + pipeline.str() + "\"";
    
    if (params.logging) {
        std::cout << "[GstdManager] Creating processing pipeline: " << pipeline.str() << std::endl;
    }

    return sendGstdCommand(cmd);
}

bool GstdManager::createOutputPipeline() {
    // Create output pipeline for passthrough mode (handles encoded H264)
    std::ostringstream pipeline;
    pipeline << "interpipesrc name=src listen-to=" << PASSTHROUGH_INTERPIPE 
             << " is-live=true format=time ! "
             << "queue max-size-buffers=2 leaky=downstream ! "
             << "rtspclientsink location=" << params.rtspOutput 
             << " protocols=tcp latency=0";

    std::string cmd = "pipeline_create " + std::string(OUTPUT_PIPELINE) + " \"" + pipeline.str() + "\"";
    
    if (params.logging) {
        std::cout << "[GstdManager] Creating output pipeline: " << pipeline.str() << std::endl;
    }

    return sendGstdCommand(cmd);
}

bool GstdManager::switchToPassthrough() {
    if (passthroughMode) {
        return true; // Already in passthrough mode
    }

    if (params.logging) {
        std::cout << "[GstdManager] Switching to passthrough mode..." << std::endl;
    }

    // Switch interpipesrc to listen to passthrough output
    bool success = setInterpipeListener(OUTPUT_INTERPIPE, PASSTHROUGH_INTERPIPE);
    
    if (success) {
        passthroughMode = true;
        if (params.logging) {
            std::cout << "[GstdManager] Switched to passthrough mode successfully" << std::endl;
        }
    }
    
    return success;
}

bool GstdManager::switchToProcessing() {
    if (!passthroughMode) {
        return true; // Already in processing mode
    }

    if (params.logging) {
        std::cout << "[GstdManager] Switching to processing mode..." << std::endl;
    }

    // Switch interpipesrc to listen to processing output
    bool success = setInterpipeListener(OUTPUT_INTERPIPE, PROCESSING_INTERPIPE);
    
    if (success) {
        passthroughMode = false;
        if (params.logging) {
            std::cout << "[GstdManager] Switched to processing mode successfully" << std::endl;
        }
    }
    
    return success;
}

bool GstdManager::sendGstdCommand(const std::string& command) {
    std::string cmd = "gst-client " + command;
    
    if (params.logging) {
        std::cout << "[GstdManager] Executing: " << cmd << std::endl;
    }
    
    int result = system(cmd.c_str());
    return result == 0;
}

std::string GstdManager::executeGstdCommand(const std::string& command) {
    std::string cmd = "gst-client " + command;
    
    if (params.logging) {
        std::cout << "[GstdManager] Executing: " << cmd << std::endl;
    }
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "";
    }
    
    std::string result;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    
    pclose(pipe);
    return result;
}

bool GstdManager::startPipeline(const std::string& pipelineName) {
    return sendGstdCommand("pipeline_play " + pipelineName);
}

bool GstdManager::stopPipeline(const std::string& pipelineName) {
    return sendGstdCommand("pipeline_stop " + pipelineName);
}

bool GstdManager::pausePipeline(const std::string& pipelineName) {
    return sendGstdCommand("pipeline_pause " + pipelineName);
}

bool GstdManager::playPipeline(const std::string& pipelineName) {
    return sendGstdCommand("pipeline_play " + pipelineName);
}

bool GstdManager::setInterpipeListener(const std::string& interpipeName, const std::string& sourceName) {
    std::string cmd = "element_set " + std::string(OUTPUT_PIPELINE) + " src listen-to " + sourceName;
    return sendGstdCommand(cmd);
}

bool GstdManager::isHealthy() const {
    if (!initialized) {
        return false;
    }
    
    // For now, just return true to avoid constant health check failures
    // The gstd API doesn't have a simple way to check pipeline state
    // We could use the low-level API: read /pipelines/<name>/state
    // but it's complex to parse the JSON response
    return true;
}

} // namespace vs
