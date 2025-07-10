# GStreamer Daemon Interpipe Streaming System

## Overview

This system implements an ultra-low latency video streaming solution using GStreamer Daemon (gstd) and interpipes. It allows seamless switching between two modes:

1. **Passthrough Mode**: Direct RTSP-to-RTSP forwarding with zero additional processing latency
2. **Processing Mode**: Full video processing pipeline with minimal additional latency

## Key Features

- **Zero-latency passthrough**: Direct encoded stream forwarding
- **Seamless mode switching**: Switch between passthrough and processing without client disconnection
- **Real-time configuration**: Change modes by modifying the config file
- **Persistent output stream**: Single RTSP output endpoint
- **Hardware acceleration**: NVIDIA GPU encoding/decoding support

## Architecture

```
Input RTSP Stream
       |
   ┌───▼────┐
   │  gstd  │
   │ daemon │
   └───┬────┘
       |
   ┌───▼────────────────┐
   │   Three Pipelines  │
   │                    │
   │ 1. Passthrough ────┼──► interpipesink (passthrough_out)
   │    (encoded H.264) │
   │                    │
   │ 2. Processing ─────┼──► interpipesink (processing_out)  
   │    (decoded BGR)   │                    │
   │                    │                    ▼
   │ 3. Output ─────────┼──► interpipesrc ──► RTSP Output
   │    (switchable)    │    (switchable)
   └────────────────────┘
```

## Installation

### Prerequisites

1. **GStreamer Daemon**:
   ```bash
   sudo apt-get install gstreamer1.0-daemon
   ```

2. **GStreamer Interpipe Plugin**:
   ```bash
   # Install from RidgeRun repository
   git clone https://github.com/RidgeRun/gst-interpipe.git
   cd gst-interpipe
   ./autogen.sh --libdir /usr/lib/aarch64-linux-gnu/
   make
   sudo make install
   ```

3. **NVIDIA GStreamer Plugins** (for Jetson):
   - Usually pre-installed on Jetson devices
   - Includes nvv4l2decoder, nvvidconv

### Building

```bash
cd /home/orin/Developer/libtracker
chmod +x examples/build_gstd_interpipe.sh
./examples/build_gstd_interpipe.sh
```

## Usage

### Basic Usage

```bash
./examples/main-gstd-interpipe config.yaml rtsp://localhost:8554/output
```

### Configuration File

Edit `examples/config_gstd_interpipe.yaml`:

```yaml
# Input source
video_source: "rtsp://192.168.1.100:8554/camera1"

# Processing modes (set to false for passthrough)
mode:
  enhancer_enabled: false      # Image enhancement
  roll_correction_enabled: false  # Roll correction
  stabilizer_enabled: false   # Video stabilization  
  tracker_enabled: false      # Object tracking
```

### Testing

1. **Run the streaming system**:
   ```bash
   ./examples/test_gstd_interpipe.sh
   ```

2. **View the output stream**:
   ```bash
   ./examples/test_rtsp_viewer.sh
   ```

3. **Manual RTSP viewing**:
   ```bash
   # Using GStreamer
   gst-launch-1.0 playbin uri=rtsp://127.0.0.1:8554/forwarded
   
   # Using VLC
   vlc rtsp://127.0.0.1:8554/forwarded
   
   # Using FFplay
   ffplay rtsp://127.0.0.1:8554/forwarded
   ```

## Mode Switching

### Runtime Mode Switching

1. **Via Configuration File**:
   - Edit the config file and save
   - The system automatically detects changes and switches modes

2. **Via Keyboard** (when not optimizing FPS):
   - Press 'p' to toggle between passthrough and processing modes
   - Press 'q' or ESC to quit

### Example Mode Switch

1. **Start in passthrough mode** (all processing disabled):
   ```yaml
   mode:
     enhancer_enabled: false
     roll_correction_enabled: false
     stabilizer_enabled: false
     tracker_enabled: false
   ```

2. **Switch to processing mode** (enable some processing):
   ```yaml
   mode:
     enhancer_enabled: true
     roll_correction_enabled: false
     stabilizer_enabled: true
     tracker_enabled: false
   ```

The system will automatically switch the interpipe source without interrupting the output stream.

## Latency Performance

- **Passthrough Mode**: ~10-20ms additional latency (network + minimal processing)
- **Processing Mode**: ~50-100ms additional latency (includes decoding, processing, encoding)

## Troubleshooting

### Common Issues

1. **"gstd daemon already running"**:
   ```bash
   pkill -f gstd
   # Then restart the application
   ```

2. **"interpipe plugin not found"**:
   ```bash
   gst-inspect-1.0 interpipesrc
   # Should show plugin info, if not, reinstall gst-interpipe
   ```

3. **"Failed to create output pipeline"**:
   - Check if RTSP server is running at the output URL
   - Verify network connectivity

4. **No video output**:
   - Check input RTSP stream is accessible
   - Verify NVIDIA decoder is working: `gst-inspect-1.0 nvv4l2decoder`

### Debug Commands

1. **Check gstd status**:
   ```bash
   gst-client pipeline_list
   ```

2. **Monitor pipeline states**:
   ```bash
   gst-client pipeline_get PIPELINE_NAME state
   ```

3. **View pipeline graphs**:
   ```bash
   gst-client pipeline_get PIPELINE_NAME graph
   ```

## API Reference

### GstdManager Methods

- `initialize()`: Start gstd and create pipelines
- `start()`: Start all pipelines
- `switchToPassthrough()`: Switch to passthrough mode
- `switchToProcessing()`: Switch to processing mode
- `stop()`: Stop all pipelines
- `cleanup()`: Clean up resources

### Pipeline Names

- `passthrough_pipeline`: Direct RTSP forwarding
- `processing_pipeline`: Decode to raw frames
- `output_pipeline`: Encode and send to output RTSP

### Interpipe Names

- `passthrough_out`: Encoded H.264 stream
- `processing_out`: Raw BGR frames
- Output pipeline switches between these sources

## Performance Tuning

### For Maximum Performance

1. **Passthrough Mode Configuration**:
   ```yaml
   mode:
     optimize_fps: true
     enhancer_enabled: false
     roll_correction_enabled: false
     stabilizer_enabled: false
     tracker_enabled: false
   ```

2. **Hardware Optimization**:
   - Use NVIDIA GPU memory pools
   - Minimize buffer sizes
   - Enable drop-on-latency

### For Quality vs Latency

1. **Balanced Configuration**:
   ```yaml
   mode:
     optimize_fps: false
     enhancer_enabled: true
     stabilizer_enabled: true
   ```

2. **Quality-focused Configuration**:
   ```yaml
   enhancer:
     enable_clahe: true
     enable_denoise: true
   ```

## Development

### Adding New Processing Modules

1. Implement the processing interface
2. Add to the processing pipeline in `main-gstd-interpipe.cpp`
3. Add configuration parameters to the YAML file
4. Test mode switching functionality

### Extending Interpipe Functionality

1. Add new interpipe sinks for different processing paths
2. Modify the output pipeline to handle multiple sources
3. Implement switching logic in `GstdManager`

## License

This project is part of the libtracker video processing library.
