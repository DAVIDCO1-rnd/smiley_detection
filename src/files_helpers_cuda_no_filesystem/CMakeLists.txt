get_filename_component(target_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(main_include_folder "${CMAKE_SOURCE_DIR}/includes")


file(GLOB SRC_FILES
		"${CMAKE_CURRENT_SOURCE_DIR}/*.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.vert"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.frag"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.tcs"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.tes"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.gs"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cs"
)

add_library(${target_name} STATIC
    ${SRC_FILES}
)

#target_link_libraries(${target_name} PUBLIC
#    opencv_utils_for_debugging
#)

target_compile_features(${target_name} PUBLIC 
cxx_std_17
)

target_compile_options(${target_name} PRIVATE /std:c++17 /MP)
target_link_options(${target_name} PUBLIC /ignore:4099)

#set_target_properties(${target_name} PROPERTIES FOLDER ray_tracing_tutorials_cuda)
