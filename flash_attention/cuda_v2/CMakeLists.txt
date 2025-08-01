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

# Create test executable
add_executable(test_flash_v2 test_flash_attention_v2.cu)

# Link test with library
target_link_libraries(test_flash_v2 flash_attention_v2 ${CUDA_LIBRARIES})

# Set include directories for test
target_include_directories(test_flash_v2 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_common
)

# Add test to CTest
add_test(NAME test_flash_v2 COMMAND test_flash_v2)