cmake_minimum_required(VERSION 3.10)
project(PhotometricLossCUDA LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# If you use OpenCV for image loading:
find_package(OpenCV REQUIRED)

# Source files
set(SOURCES
    photometric_loss.cu
)

add_executable(PhotometricLoss ${SOURCES})

target_include_directories(PhotometricLoss PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(PhotometricLoss PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES})

set_target_properties(PhotometricLoss PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50;60;70;75;80"
)
