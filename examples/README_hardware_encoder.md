# Hardware Encoder Optimized Pipeline

This is an optimized version of the main-gstd-jetson.cpp that uses low-level NVIDIA Jetson Multimedia API for hardware encoding, providing the lowest latency for video processing and re-streaming.

## Key Optimizations

1. **Low-Level Hardware Encoding**: Uses NVIDIA Jetson Multimedia API directly instead of GStreamer plugins
2. **Optimized Memory Management**: Direct buffer operations with minimal copying
3. **Efficient Pipeline Switching**: Seamless switching between passthrough and processing modes
4. **Zero-Copy Operations**: Where possible, reduces memory copy operations
5. **Hardware-Accelerated Processing**: Leverages GPU for encoding operations

## Architecture

### Passthrough Mode
- Direct H.265 forwarding using GStreamer (no re-encoding)
- Minimal latency for streams that don't require processing

### Processing Mode
- Hardware-accelerated H.264 encoding using NVIDIA Video Encoder API
- Optimized for processed frames from OpenCV
- Uses EGL/CUDA interop for efficient GPU memory management

## Building

```bash
# Make sure you have the Jetson Multimedia API available
sudo apt-get install nvidia-l4t-jetson-multimedia-api

# Build the optimized version
./build_hardware_encoder.sh
```

## Usage

```bash
# Run with hardware encoder optimizations
./build_hardware_encoder/hardware_encoder_example config_hardware_encoder.yaml
```

## Configuration

The configuration file controls when to use hardware encoding:

```yaml
mode:
  # When any of these are enabled, switches to hardware encoder
  enhancer_enabled: true    # Enable for image enhancement
  roll_correction_enabled: true  # Enable for roll correction
  stabilization_enabled: true    # Enable for stabilization
  tracker_enabled: true          # Enable for object tracking
```

## Performance Benefits

1. **Lower Latency**: Direct hardware encoding reduces processing pipeline latency
2. **Better Quality**: Hardware encoder provides better rate control and quality
3. **Lower CPU Usage**: Offloads encoding to dedicated hardware
4. **Memory Efficiency**: Reduces memory bandwidth through optimized buffer management

## Hardware Requirements

- NVIDIA Jetson Orin Nano (or compatible)
- JetPack 5.0+ with Multimedia API
- CUDA-capable GPU with hardware encoder support

## Technical Details

### Hardware Encoder Pipeline
1. OpenCV frame → NvBuffer conversion
2. Hardware H.264 encoding via NvVideoEncoder
3. Encoded stream → GStreamer RTSP output

### Memory Management
- Uses EGL display for GPU memory management
- Direct DMA buffer operations
- Minimizes CPU-GPU memory transfers

### Threading
- Separate encoder capture thread for non-blocking operation
- Asynchronous buffer handling
- Efficient producer-consumer pattern

## Troubleshooting

1. **Build Issues**: Ensure Jetson Multimedia API is installed
2. **Runtime Errors**: Check that hardware encoder resources are available
3. **Performance**: Monitor GPU memory usage and encoder utilization

## Notes

- This version is specifically optimized for Jetson Orin Nano
- Hardware encoder supports H.264 output (for better compatibility)
- Passthrough mode still uses original H.265 for maximum efficiency
