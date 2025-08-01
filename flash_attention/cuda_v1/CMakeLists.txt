# Flash Attention v1 CUDA Implementation

# Create test executable
add_executable(test_flash_v1 test_flash_attention_v1.cu)

# Link with CUDA
target_link_libraries(test_flash_v1 ${CUDA_LIBRARIES})

# Set include directories for test
target_include_directories(test_flash_v1 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_common
)

# Add test to CTest
add_test(NAME test_flash_v1 COMMAND test_flash_v1)