add_executable(path_linalg_test test.cu)

set_property(TARGET path_linalg_test PROPERTY CUDA_ARCHITECTURES native)

set_property(TARGET path_linalg_test
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(path_linalg_test PRIVATE path_linalg)