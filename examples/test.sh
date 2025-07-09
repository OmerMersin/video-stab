#!/bin/bash

# Ensure that a parameter is passed
if [ -z "$1" ]; then
    echo "Usage: ./test.sh <source-file-name>"
    exit 1
fi

FILE_NAME=$1

# Navigate to the build directory
cd ~/Developer/libtracker/build || exit

# Clean and rebuild the project
cmake .. -DBUILD_SHARED_LIBS=ON -D WITH_CUDA=ON
make -j$(nproc)
sudo make install

# Navigate to the examples directory
cd ../examples || exit

# Set DeepStream paths
DEEPSTREAM_INCLUDE="/opt/nvidia/deepstream/deepstream/sources/includes"
DEEPSTREAM_LIB="/opt/nvidia/deepstream/deepstream/lib"

# Set NVIDIA Jetson Multimedia API paths
JETSON_MULTIMEDIA_API_PATH="/usr/src/jetson_multimedia_api"
JETSON_MULTIMEDIA_INCLUDE="$JETSON_MULTIMEDIA_API_PATH/include"
JETSON_MULTIMEDIA_SAMPLES="$JETSON_MULTIMEDIA_API_PATH/samples"
JETSON_MULTIMEDIA_CLASSES="$JETSON_MULTIMEDIA_SAMPLES/common/classes"

# Set CUDA paths
CUDA_PATH="/usr/local/cuda"
CUDA_LIB64="$CUDA_PATH/lib64"

# Compile the specified source file with NVIDIA Jetson Multimedia API support
g++ "$FILE_NAME.cpp" \
  "$JETSON_MULTIMEDIA_CLASSES/NvVideoEncoder.cpp" \
  "$JETSON_MULTIMEDIA_CLASSES/NvVideoDecoder.cpp" \
  "$JETSON_MULTIMEDIA_CLASSES/NvV4l2Element.cpp" \
  "$JETSON_MULTIMEDIA_CLASSES/NvV4l2ElementPlane.cpp" \
  "$JETSON_MULTIMEDIA_CLASSES/NvElement.cpp" \
  "$JETSON_MULTIMEDIA_CLASSES/NvElementProfiler.cpp" \
  "$JETSON_MULTIMEDIA_CLASSES/NvLogging.cpp" \
  "$JETSON_MULTIMEDIA_CLASSES/NvBuffer.cpp" \
  "$JETSON_MULTIMEDIA_CLASSES/NvUtils.cpp" \
  -o "$FILE_NAME" \
  -I/usr/local/include \
  -I"$JETSON_MULTIMEDIA_INCLUDE" \
  -I"$JETSON_MULTIMEDIA_SAMPLES/common/classes" \
  -I"$JETSON_MULTIMEDIA_SAMPLES/common/algorithm/trt" \
  -I"/usr/include/libdrm" \
  -I"/usr/include/EGL" \
  -I"/usr/include/GLES2" \
  -DUSE_NVBUF_TRANSFORM_API \
  $(pkg-config --cflags --libs opencv4) \
  $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0 || echo "-L/usr/lib/aarch64-linux-gnu -lgstreamer-1.0 -lgstrtspserver-1.0 -lgstapp-1.0") \
  $(pkg-config --cflags --libs glib-2.0 gobject-2.0) \
  $(pkg-config --cflags --libs libcurl || echo "-lcurl") \
  $(pkg-config --cflags --libs jsoncpp) \
  -I"$DEEPSTREAM_INCLUDE" -L"$DEEPSTREAM_LIB" -lnvds_meta \
  -L/usr/local/lib \
  -L"$CUDA_LIB64" \
  -lvideo-stab -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_videoio \
  -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudawarping -lopencv_calib3d \
  -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_cudaarithm -lopencv_photo -lopencv_features2d \
  -lgstreamer-1.0 -lgstrtspserver-1.0 -lgstapp-1.0 -lgobject-2.0 -lglib-2.0 -pthread \
  -lX11 -lXext -lXfixes -lXrender -lXrandr -lXi -lXcomposite -lXcursor -lXdamage \
  -lnvdsgst_meta -lnvds_meta -lnvds_utils \
  -L/usr/lib/aarch64-linux-gnu/tegra \
  -lnvv4l2 -lEGL -lGLESv2 \
  -lnvjpeg -lnvosd \
  -lnvbufsurface -lnvbufsurftransform \
  -Wl,--no-as-needed -ljpeg -ldrm -lv4l2 \
  -Wl,-rpath,/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64



# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful! Run ./$FILE_NAME to execute."
else
    echo "❌ Compilation failed!"
    exit 1
fi

