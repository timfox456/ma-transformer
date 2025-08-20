cmake_minimum_required(VERSION 3.14)
project(ma_transformer_tests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/csrc)

# Add source files
set(CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/csrc/ma_core.cpp
)

# Create test executable
add_executable(ma_core_tests
    test_ma_core.cpp
    ${CORE_SOURCES}
)

# Link with Google Test
target_link_libraries(ma_core_tests
    GTest::gtest
    GTest::gtest_main
)

# Add test discovery
include(GoogleTest)
gtest_discover_tests(ma_core_tests)