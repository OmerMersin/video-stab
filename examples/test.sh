#!/bin/bash

# Ensure that a parameter is passed
if [ -z "$1" ]; then
    echo "Usage: ./test.sh <source-file-name>"
    exit 1
fi

FILE_NAME=$1

# Navigate to the build directory
cd ~/Developer/StabC/v2/libvideo-stab/build || exit

# Clean and rebuild the project
cmake .. -DBUILD_SHARED_LIBS=ON -D WITH_CUDA=ON
make -j$(nproc)
sudo make install

# Navigate to the examples directory
cd ../examples || exit

# Compile the specified source file
g++ "$FILE_NAME.cpp" -o "$FILE_NAME" \
  $(pkg-config --cflags --libs opencv4) \
  $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0 || echo "-L/usr/lib/aarch64-linux-gnu -lgstreamer-1.0 -lgstrtspserver-1.0 -lgstapp-1.0") \
  $(pkg-config --cflags --libs glib-2.0 gobject-2.0) \
  -L/usr/local/lib \
  -lvideo-stab -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_videoio \
  -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudawarping -lopencv_calib3d \
  -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_cudaarithm -lopencv_photo \
  -lgstreamer-1.0 -lgstrtspserver-1.0 -lgstapp-1.0 -lgobject-2.0 -lglib-2.0 -pthread -lX11


# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful! Run ./$FILE_NAME to execute."
else
    echo "❌ Compilation failed!"
    exit 1
fi

