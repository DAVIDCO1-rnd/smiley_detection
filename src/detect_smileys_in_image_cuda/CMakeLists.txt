get_filename_component(target_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Find CUDA
find_package(CUDA REQUIRED)

set(main_includes_folder "${CMAKE_SOURCE_DIR}/include")

file(GLOB SRC_FILES
		"${CMAKE_CURRENT_SOURCE_DIR}/*.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.geo"
)



file(GLOB HEADERS_FILES "${includes_folder}/*.h")
file(GLOB SOURCES_FILES "${sources_folder}/*.cpp")
add_executable(${target_name} 
	${SRC_FILES}
)

target_include_directories(${target_name} PUBLIC
	${CMAKE_SOURCE_DIR}/src/files_helpers_cuda_no_filesystem
)

enable_language(CUDA)
set_target_properties(${target_name} PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50"
)

target_link_libraries(${target_name} PUBLIC 
    "$<IF:$<CONFIG:Debug>,${opencv_libs_debug_files},${OpenCV_LIBS}>"
	cudart
	files_helpers_cuda_no_filesystem
)

target_compile_features(${target_name} PUBLIC cxx_std_17)




