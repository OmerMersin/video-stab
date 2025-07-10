cmake_minimum_required(VERSION 3.10)
project(video-stab VERSION 1.0 LANGUAGES CXX CUDA)

# Use C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------
# Find OpenCV
# --------------------------------------------------
find_package(OpenCV REQUIRED)

# --------------------------------------------------
# Find GStreamer + RTSP Server (via pkg-config)
# --------------------------------------------------
find_package(PkgConfig REQUIRED)

# If you'd like to check exact versions, use:
# pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.14 gstreamer-rtsp-server-1.0>=1.14)
# But here, we just require them with no version constraints:
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0)

pkg_check_modules(GLIB REQUIRED glib-2.0 gobject-2.0)
include_directories(${GLIB_INCLUDE_DIRS})

link_directories(
  /opt/nvidia/deepstream/deepstream-7.1/lib
)
# --------------------------------------------------
# Create the library
# --------------------------------------------------
add_library(video-stab STATIC
    src/Stabilizer.cpp
    src/CamCap.cpp
    src/RollCorrection.cpp
    src/AutoZoomCrop.cpp
    src/Enhancer.cpp
    src/RTSPServer.cpp
    src/DeepStreamTracker.cpp
    src/TcpReciever.cpp
    src/GstdManager.cpp
    src/CamCapInterpipe.cpp
)

# Public headers
target_include_directories(video-stab
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${GST_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        /opt/nvidia/deepstream/deepstream-7.1/sources/includes
)

# Link libraries
target_link_libraries(video-stab
    PRIVATE
        ${OpenCV_LIBS}
        ${GST_LIBRARIES}
        ${GLIB_LIBRARIES}
        nvdsgst_meta
        nvds_meta
        nvds_inferutils
        nvds_infercustomparser
        pthread
        nvbufsurface
        nvbufsurftransform

)

# Optionally add definitions or compile options if needed:
# target_compile_definitions(video-stab PRIVATE ${GST_CFLAGS_OTHER})

# If you want to integrate CUDA more explicitly, you can set:
# set_target_properties(video-stab PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# or
# target_compile_options(video-stab PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_53>)

# --------------------------------------------------
# Installation instructions
# --------------------------------------------------
install(TARGETS video-stab DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/video-stabConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/video-stabConfigVersion.cmake"
    DESTINATION
        lib/cmake/video-stab
)
