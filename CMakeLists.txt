####
# Set minimum version of CMake. Since command 'project' use
# VERSION sub-option we need at least 3.0.
# Note: If you use 2.6 or 2.4, God kills a kitten. Seriously.
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

####
# Set variables:
#   * PROJECT_NAME
#   * PROJECT_VERSION
project(tiny_dnn VERSION 1.0.0 LANGUAGES C CXX)

#####
# Enables link_directories() treat paths relative
# to the source dir.
if(POLICY CMP0015)
    cmake_policy(SET CMP0015 NEW)
endif(POLICY CMP0015)

#####
# Enables project() command manages VERSION variables.
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

#####
# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("==========================================================================================")
        message(STATUS "Build type: Debug. Performance will be terrible!")
        message(STATUS "Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
        message("==========================================================================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

####
# Define user options
option(USE_SSE        "Build tiny-dnn with SSE library support"     ON)
option(USE_AVX        "Build tiny-dnn with AVX library support"     ON)
option(USE_AVX2       "Build tiny-dnn with AVX2 library support"   OFF)
option(USE_TBB        "Build tiny-dnn with TBB library support"    OFF)
option(USE_OMP        "Build tiny-dnn with OMP library support"    OFF)
option(USE_NNPACK     "Build tiny-dnn with NNPACK library support" OFF)
option(USE_OPENCL     "Build tiny-dnn with OpenCL library support" OFF) 
option(USE_LIBDNN     "Build tiny-dnn with GreenteaLibDNN library support" OFF)
option(USE_SERIALIZER "Build tiny-dnn with Serialization support"   ON)
option(USE_DOUBLE     "Build tiny-dnn with double precision computations"  OFF)
option(USE_IMAGE_API  "Build tiny-dnn with Image API support"       ON)
option(USE_GEMMLOWP   "Build tiny-dnn with gemmlowp support"       OFF)

option(BUILD_TESTS      "Set to ON to build tests"              OFF)
option(BUILD_EXAMPLES   "Set to ON to build examples"           OFF)
option(BUILD_DOCS       "Set to ON to build documentation"      OFF)
option(BUILD_BENCHMARKS "Set to ON to build benchmarks"         OFF)
option(COVERALLS        "Set to ON to build with code coverage" OFF)
option(PROFILE          "Set to ON to build with profiling"     OFF)
option(USE_ASAN         "Set to ON to build with ASan"          OFF)

####
# Create the library target

set(project_library_target_name ${PROJECT_NAME})
set(PACKAGE_NAME TinyDNN)

add_library(${project_library_target_name} INTERFACE)

####
# Setup the optional dependencies

# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Tiny-dnn provides a couple of multithreading solutions.
# The user can specify to use Intel Threading Building Blocks (TBB)
# or Open Multi-Processing (OpenMP) as a backend for multi threading
# processing. In case that none of this libraries are required, tiny-dnn
# will use the standard C++11 Thread support library.

# Find Intel Threading Building Blocks (TBB)
find_package(TBB QUIET)
if(USE_TBB AND TBB_FOUND)
    message(STATUS "Found Intel TBB: ${TBB_INCLUDE_DIR}")
    # In case that TBB is found we force to disable OpenMP since
    # tiny-dnn does not support mutiple multithreading backends.
    set(USE_OMP OFF)
    #TODO: add definitions in configure
    add_definitions(-DCNN_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
    list(APPEND REQUIRED_LIBRARIES ${TBB_LIBRARIES})
elseif(USE_TBB AND NOT TBB_FOUND)
    # In case the user sets the flag USE_TBB to ON, the CMake build-tree
    # will require to find TBB in your system. Otherwise, the user can
    # set the paths to headers and libs by hand.
    message(FATAL_ERROR "Intel TBB not found. Please set TBB_INCLUDE_DIRS & "
            "TBB_LIBRARIES")
endif()

if(NOT USE_SERIALIZER)
    add_definitions(-DCNN_NO_SERIALIZATION)
endif()

if(USE_DOUBLE)
    add_definitions(-DCNN_USE_DOUBLE)
endif()

if(USE_IMAGE_API)
    add_definitions(-DDNN_USE_IMAGE_API)
endif()

if(USE_GEMMLOWP)
    add_definitions(-DUSE_GEMMLOWP)
endif()

# Find Open Multi-Processing (OpenMP)
find_package(OpenMP QUIET)
if(USE_OMP AND OPENMP_FOUND)
    message(STATUS "Found OpenMP")
    # In case that OMP is found we force to disable Intel TBB since
    # tiny-dnn does not support mutiple multithreading backends.
    set(USE_TBB OFF)
    add_definitions(-DCNN_USE_OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
elseif(USE_OMP AND NOT OPENMP_FOUND)
    # In case the user sets the flag USE_OMP to ON, the CMake build-tree
    # will require to find OMP in your system. Otherwise, the user can
    # set the CMAKE_C_FLAGS and CMAKE_CXX_FLAGS by hand.
    message(FATAL_ERROR "Can't find OpenMP. Please set OpenMP_C_FLAGS & "
            "OpenMP_CXX_FLAGS")
endif()

# Find NNPACK: Acceleration package for neural networks on multi-core CPUs
find_package(NNPACK QUIET)
if(USE_NNPACK AND NNPACK_FOUND)
    message(STATUS "Found NNPACK: ${NNPACK_INCLUDE_DIR}")
    add_definitions(-DCNN_USE_NNPACK)
    include_directories(SYSTEM ${NNPACK_INCLUDE_DIR})
    include_directories(SYSTEM ${NNPACK_INCLUDE_DIR}/../deps/pthreadpool/include)
    list(APPEND REQUIRED_LIBRARIES ${NNPACK_LIB})
    list(APPEND REQUIRED_LIBRARIES ${PTHREADPOOL_LIB})
elseif(USE_NNPACK AND NOT NNPACK_FOUND)
    # In case the user sets the flag USE_NNPACK to ON, the CMake build-tree
    # will require to find NNPACK in your system. Otherwise, the user can
    # set the paths to headers and libs by hand.
    message(FATAL_ERROR "Can't find NNPACK. Please set NNPACK_INCLUDE_DIR "
            " & NNPACK_LIB")
endif()

# in case that TBB and OMP are not enabled/found,
# we enable standard C++11 multithread support.
if((NOT USE_TBB) AND (NOT USE_OMP) AND (NOT WIN32))
    #list(APPEND EXTRA_C_FLAGS -pthread)
    set(USE_PTHREAD ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    message(STATUS "TBB and OMP disabled: Using Pthread instead.")
else((NOT USE_TBB) AND (NOT USE_OMP))
    set(USE_PTHREAD OFF)
endif((NOT USE_TBB) AND (NOT USE_OMP) AND (NOT WIN32))

find_package(OpenCL QUIET)
if(USE_OPENCL AND OpenCL_FOUND)
    message(STATUS "Found OpenCL: ${OpenCL_INCLUDE_DIRS}")
    #add_definitions(-DCNN_HAVE_OPENCL)
    add_definitions(-DUSE_OPENCL)
    include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
    list(APPEND REQUIRED_LIBRARIES ${OpenCL_LIBRARY})
elseif(USE_OPENCL AND NOT OpenCL_FOUND)
    # In case the user sets the flag USE_OPENCL to ON, the CMake build-tree
    # will require to find OPENCL in your system. Otherwise, the user can
    # set the paths to headers and libs by hand.
    message(FATAL_ERROR "Can't find OpenCL.")
endif()

find_package(GreenteaLibDNN QUIET)
if(OpenCL_FOUND AND USE_LIBDNN AND GreenteaLibDNN_FOUND)
    message(STATUS "Found GreenteaLibDNN: ${GREENTEA_INCLUDE_DIRS}")
    add_definitions(-DCNN_USE_LIBDNN)
    include_directories(SYSTEM ${GREENTEA_INCLUDE_DIRS})
    list(APPEND REQUIRED_LIBRARIES greentea_libdnn ${GREENTEA_LIBRARIES})
elseif(USE_LIBDNN AND NOT OpenCL_FOUND)
    message(FATAL_ERROR "OpenCL is needed for GreenteaLibDNN.")
elseif(USE_LIBDNN AND NOT LIBDNN_FOUND)
    # In case the user sets the flag USE_LIBDNN to ON, the CMake build-tree
    # will require to find LibDNN in your system. Otherwise, the user can
    # set the paths to headers and libs by hand.
    message(FATAL_ERROR "Can't find LibDNN.")
endif()

find_package(HDF5 QUIET)
if(HDF5_FOUND AND USE_SERIALIZER)
    message(STATUS "Found HDF C++ API Modules: ${HDF5_INCLUDE_DIRS}")
    link_directories( ${HDF5_LIBRARY_DIRS} )
    include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} )
    list(APPEND REQUIRED_LIBRARIES ${HDF5_LIBRARIES} hdf5_cpp)
endif()

####
# Setup the compiler options

# set c++ standard to c++14.
# Note: not working on CMake 2.8. We assume that user has
#       a compiler with C++14 support.

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++14 support has been enabled by default.")

include(cotire)

# Unix
if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-msse3" COMPILER_HAS_SSE_FLAG)
    check_cxx_compiler_flag("-mavx"  COMPILER_HAS_AVX_FLAG)
    check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2_FLAG)
    check_cxx_compiler_flag("-mfma" COMPILER_HAS_AVX2_FLAG)

    # set Streaming SIMD Extension (SSE) instructions
    if(USE_SSE AND COMPILER_HAS_SSE_FLAG)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
    endif(USE_SSE AND COMPILER_HAS_SSE_FLAG)
    # set Advanced Vector Extensions (AVX)
    if(USE_AVX AND COMPILER_HAS_AVX_FLAG)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx")
    endif(USE_AVX AND COMPILER_HAS_AVX_FLAG)
    # set Advanced Vector Extensions 2 (AVX2)
    if(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx2 -mfma -march=core-avx2")
    endif(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)

    # include extra flags to the compiler
    # TODO: add info about those flags.
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing -Wno-deprecated")
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pthread")
elseif(MSVC)
    if(USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:SSE2")
    endif(USE_SSE)
    if(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX")
    endif(USE_AVX)
    if(USE_AVX2)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX2")
    endif(USE_AVX2)
    # include specific flags for release and debug modes.
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}
        /Ox /Oi /Ot /Oy /GL /fp:fast /GS-")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /W4 /bigobj")
    # this is fine
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
    add_definitions(-D _SCL_SECURE_NO_WARNINGS)
    # prolly powerless with header-only project
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /MP")
endif()

####
# Set compiler options
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")

if(PROFILE AND (CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -pg")
endif()

if (USE_ASAN)
    #enable ASan
    set(ENV{ASAN_OPTIONS} "strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:symbolize=1")
    CHECK_CXX_COMPILER_FLAG(-fsanitize-address-use-after-scope HasUseAfterScope)
    set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
    if (HasUseAfterScope)
        set(ASAN_FLAGS "${ASAN_FLAGS} -fsanitize-address-use-after-scope")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
endif()
####
# Write the config.h
# TODO: replace for tiny_dnn/config.h
# configure_file(cmake/Templates/tinydnn_config.h.in
#               "${PROJECT_BINARY_DIR}/tinydnn_config.h")

####
# Setup the cmake config files
string(REGEX REPLACE "_" "" PROJECT_NAME_JOINED ${PROJECT_NAME})

set(cmake_conf_file         "${PROJECT_NAME_JOINED}-config.cmake")
set(cmake_conf_version_file "${PROJECT_NAME_JOINED}-config-version.cmake")
set(cmake_targets_file      "${PROJECT_NAME_JOINED}-targets.cmake")

set(targets_export_name "${PROJECT_NAME_JOINED}-targets")
set(namespace "${PACKAGE_NAME}::")

# Set up install directories. INCLUDE_INSTALL_DIR and
# CMAKECONFIG_INSTALL_DIR must not be absolute paths.
if(WIN32)
    set(include_install_dir Include)
    set(include_install_dir_full Include)
    set(config_install_dir CMake)
elseif(UNIX)
    set(include_install_dir include)
    set(include_install_dir_postfix "${project_library_target_name}")
    set(include_install_dir_full    "${include_install_dir}/${include_install_dir_postfix}")

    set(config_install_dir share/${PACKAGE_NAME})
else()
    message(FATAL_ERROR "Not supported system type. Options: UNIX or WIN32.")
endif()

# configure the library target
target_include_directories(
    ${project_library_target_name} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${include_install_dir_full}>)

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Templates/cmake-uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake-uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake-uninstall.cmake)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Templates/${cmake_conf_file}.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${cmake_conf_file}"
    PATH_VARS include_install_dir_full
    INSTALL_DESTINATION ${config_install_dir})

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${cmake_conf_version_file}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
# Create *-targets.cmake file for build directory
install(TARGETS ${project_library_target_name}
        EXPORT  ${targets_export_name}
        INCLUDES DESTINATION ${include_install_dir})

export(EXPORT ${targets_export_name}
       FILE   ${CMAKE_CURRENT_BINARY_DIR}/${cmake_targets_file})

# Install *-targets.cmake file
install(EXPORT      ${targets_export_name}
        NAMESPACE   ${namespace}
        DESTINATION ${config_install_dir})

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${cmake_conf_file}"
    "${CMAKE_CURRENT_BINARY_DIR}/${cmake_conf_version_file}"
    "${PROJECT_SOURCE_DIR}/cmake/Modules/FindTBB.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/Modules/FindNNPACK.cmake"
    DESTINATION ${config_install_dir} COMPONENT cmake)

# Install headers
install(DIRECTORY   ${PROJECT_SOURCE_DIR}/${project_library_target_name}
        DESTINATION ${include_install_dir})

# Check if protobuf available
include(cmake/protoc.cmake)

# Subdirectories for examples, testing and documentation
# TODO: explain in brief about different examples, test and docs.
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)

if(BUILD_TESTS)
    add_subdirectory(test)
endif(BUILD_TESTS)

if(BUILD_DOCS)
    add_subdirectory(docs)
endif(BUILD_DOCS)

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

####
# Configuration summary
include(cmake/summary.cmake)
tinydnn_print_configuration_summary()


