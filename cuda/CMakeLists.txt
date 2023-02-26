cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(path_linalg LANGUAGES CXX CUDA)

cmake_policy(SET CMP0076 NEW)

include(CTest)

add_library(path_linalg STATIC)

target_include_directories(path_linalg PUBLIC include)
set_property(TARGET path_linalg PROPERTY CUDA_ARCHITECTURES native)

add_subdirectory(src)
add_subdirectory(test)

# Request that path_linalg be built with -std=c++11
# As this is a public compile feature anything that links to
# path_linalg will also build with -std=c++11
target_compile_features(path_linalg PUBLIC cxx_std_20)

# We need to explicitly state that we need all CUDA files in the
# path_linalg library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(path_linalg
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
