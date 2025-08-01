# Flash Attention v2 CUDA Implementation

set(CUDA_V2_SOURCES
    flash_attention_v2.cu
    flash_attention_v2_kernel.cu
    test_flash_attention_v2.cu
)

# Create library
add_library(flash_attention_v2 STATIC ${CUDA_V2_SOURCES})

# Link with CUDA
target_link_libraries(flash_attention_v2 ${CUDA_LIBRARIES})

# Set include directories
target_include_directories(flash_attention_v2 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_common
)

# Set properties
set_target_properties(flash_attention_v2 PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Create basic CUDA test executable (C++ only)
set_source_files_properties(test_basic_cuda.cpp PROPERTIES LANGUAGE CXX)
add_executable(test_flash_v2 test_basic_cuda.cpp)
target_link_libraries(test_flash_v2 ${CUDA_LIBRARIES})

# Add test to CTest
add_test(NAME test_flash_v2 COMMAND test_flash_v2)