include(CTest)
enable_testing()


include(../cmake/DownloadProject/DownloadProject.cmake)
#set(gtest_disable_pthreads on) #TODO(randl): Windows?
download_project(
    PROJ googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG master
    UPDATE_DISCONNECTED 1
)

add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

enable_testing()

# in ../googletest-src/googletest/CMakeLists.txt, BUILD_SHARED_LIBS is set to OFF
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /MT")
endif()

add_executable(print_device print_device.cpp)
target_link_libraries(print_device
    ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(tiny_dnn_test test.cpp test_no_duplicate_symbols.cpp)
target_link_libraries(tiny_dnn_test
    ${project_library_target_name} ${REQUIRED_LIBRARIES} gtest gmock)

add_test(all_tests tiny_dnn_test)
# workaround for https://gitlab.kitware.com/cmake/cmake/issues/8774
add_custom_target(run_tests COMMAND ${CMAKE_CTEST_COMMAND}
    DEPENDS tiny_dnn_test)

if(COVERALLS)
    message(STATUS "Code coverage: Enabled")
    set_target_properties(tiny_dnn_test
        PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage"
        LINK_FLAGS "-lgcov --coverage")

else(COVERALLS)
    message(STATUS "Code coverage: Disabled")
endif(COVERALLS)

if(PROTO_CPP_AVAILABLE)
    set_source_files_properties(${proto_file} PROPERTIES GENERATED TRUE)
    target_sources(tiny_dnn_test PUBLIC ${proto_file})
    set_source_files_properties(test.cpp PROPERTIES COMPILE_DEFINITIONS
                                CNN_USE_CAFFE_CONVERTER)
    target_link_libraries(tiny_dnn_test
            ${project_library_target_name}
            ${PROTOBUF_LIBRARIES} ${REQUIRED_LIBRARIES})   
endif()

if(PROTO_CPP_GENERATE)
    add_dependencies(tiny_dnn_test generated_proto)
endif()

cotire(print_device tiny_dnn_test)
