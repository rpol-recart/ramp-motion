cmake_minimum_required(VERSION 3.18)
project(ramp_motion_preproc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4.10 REQUIRED COMPONENTS core imgproc cudaimgproc cudabgsegm cudaarithm)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gobject-2.0 glib-2.0)

set(DS_ROOT /opt/nvidia/deepstream/deepstream)
set(DS_INCLUDE ${DS_ROOT}/sources/includes)
set(DS_PREPROC_INCLUDE
    ${DS_ROOT}/sources/gst-plugins/gst-nvdspreprocess/include
    ${DS_ROOT}/sources/gst-plugins/gst-nvdspreprocess/nvdspreprocess_lib)
set(DS_LIBS ${DS_ROOT}/lib)

add_library(ramp_motion_preproc SHARED
    src/custom_lib.cpp
    src/clahe_mog2.cu
    src/motion_meta.cpp
)

target_include_directories(ramp_motion_preproc PRIVATE
    ${DS_INCLUDE}
    ${DS_PREPROC_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ramp_motion_preproc PRIVATE
    CUDA::cudart
    ${OpenCV_LIBS}
    PkgConfig::GST
    -L${DS_LIBS}
    nvdsgst_helper
    nvbufsurface
    nvbufsurftransform
    nvdsgst_meta
    nvds_meta
)

set_target_properties(ramp_motion_preproc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89;90"
)
