include(../cmake/DownloadProject/DownloadProject.cmake)
download_project(
        PROJ benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.1.0
        UPDATE_DISCONNECTED 1
)

add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})
include_directories("${benchmark_SOURCE_DIR}/include")

add_executable(tiny_dnn_benchmarks benchmarks.cpp)
target_link_libraries(tiny_dnn_benchmarks
        ${project_library_target_name} ${REQUIRED_LIBRARIES} benchmark)

find_package(HDF5 REQUIRED)

#### Clang-tidy check
#include(../cmake/clang-cxx-dev-tools.cmake)
