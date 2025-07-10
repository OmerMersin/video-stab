#ifndef GSTD_MANAGER_H
#define GSTD_MANAGER_H

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>

/**
 * GstdManager - Manages GStreamer daemon pipelines for ultra-low latency streaming
 * Uses interpipes to switch between passthrough and processing modes seamlessly
 */
namespace vs {

class GstdManager {
public:
    struct Parameters {
        std::string rtspSource;           // Input RTSP source
        std::string rtspOutput;           // Output RTSP destination
        std::string codec = "h264";       // Codec (h264/h265)
        int width = 1920;                 // Frame width
        int height = 1080;                // Frame height
        int fps = 30;                     // Frame rate
        int bitrate = 4000;               // Bitrate in kbps
        bool logging = true;              // Enable logging
    };

    explicit GstdManager(const Parameters& params);
    ~GstdManager();

    // Pipeline management
    bool initialize();
    bool start();
    void stop();
    void cleanup();

    // Mode switching
    bool switchToPassthrough();
    bool switchToProcessing();
    bool isPassthroughMode() const { return passthroughMode; }

    // Status
    bool isHealthy() const;
    bool isInitialized() const { return initialized; }

    // Get interpipe names for external use
    std::string getProcessingInterpipeName() const { return "processing_input"; }
    std::string getOutputInterpipeName() const { return "output_sink"; }

private:
    // Internal pipeline management
    bool createPassthroughPipeline();
    bool createProcessingPipeline();
    bool createOutputPipeline();
    
    // gstd communication
    bool sendGstdCommand(const std::string& command);
    std::string executeGstdCommand(const std::string& command);
    
    // Pipeline control
    bool startPipeline(const std::string& pipelineName);
    bool stopPipeline(const std::string& pipelineName);
    bool pausePipeline(const std::string& pipelineName);
    bool playPipeline(const std::string& pipelineName);
    
    // Interpipe control
    bool setInterpipeListener(const std::string& interpipeName, const std::string& sourceName);

    Parameters params;
    std::atomic<bool> initialized{false};
    std::atomic<bool> passthroughMode{true};
    std::atomic<bool> healthCheckRunning{false};
    
    std::thread healthCheckThread;
    
    // Pipeline names
    static constexpr const char* PASSTHROUGH_PIPELINE = "passthrough_pipeline";
    static constexpr const char* PROCESSING_PIPELINE = "processing_pipeline";
    static constexpr const char* OUTPUT_PIPELINE = "output_pipeline";
    
    // Interpipe names
    static constexpr const char* PASSTHROUGH_INTERPIPE = "passthrough_out";
    static constexpr const char* PROCESSING_INTERPIPE = "processing_out";
    static constexpr const char* OUTPUT_INTERPIPE = "output_sink";
};

} // namespace vs

#endif // GSTD_MANAGER_H
