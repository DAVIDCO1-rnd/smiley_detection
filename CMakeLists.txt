cmake_minimum_required(VERSION 3.5)
project(smiley_detection VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(VIPER_ARCH "x64")
else(CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(VIPER_ARCH "x86")
endif()

# Define a variable with a default path
set(OPENCV_DEBUG_INSTALL_FOLDER "D:/Developments/opencv_gpu/build/install_debug" CACHE PATH "Path to the folder in which the opencv install target copies the products files to in Debug mode")


set(OPENCV_DEBUG_BIN_FOLDER "${OPENCV_DEBUG_INSTALL_FOLDER}/x64/vc16/bin")
file(GLOB opencv_dlls_debug_files "${OPENCV_DEBUG_BIN_FOLDER}/*.dll")

set(OPENCV_DEBUG_LIB_FOLDER "${OPENCV_DEBUG_INSTALL_FOLDER}/x64/vc16/lib")
file(GLOB opencv_libs_debug_files "${OPENCV_DEBUG_LIB_FOLDER}/*.lib")

set(VIPER_BINARY_DIR_OUTPUT ${CMAKE_BINARY_DIR}/bin/${VIPER_ARCH}/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${VIPER_BINARY_DIR_OUTPUT})

set(SRC_FOLDERS ${CMAKE_SOURCE_DIR}/src)

SUBDIRLIST(SUBDIRS ${SRC_FOLDERS})

FOREACH(subdir ${SUBDIRS})
  # if(${subdir} STREQUAL "ImageProcessing.Cuda.LowLevel")
  #   continue()
  # else()
  #   add_subdirectory(${SRC_FOLDERS}/${subdir})  
  # endif()

  add_subdirectory(${SRC_FOLDERS}/${subdir})
ENDFOREACH()

# add_subdirectory("${CMAKE_SOURCE_DIR}/tutorials")
# add_subdirectory("${CMAKE_SOURCE_DIR}/cmake_csharp_examples")
# add_subdirectory("${CMAKE_SOURCE_DIR}/practice_projects")
# add_subdirectory("${CMAKE_SOURCE_DIR}/vhgs_communication")
# add_subdirectory("${CMAKE_SOURCE_DIR}/examples")
# add_subdirectory("${CMAKE_SOURCE_DIR}/examples_david")
# add_subdirectory("${CMAKE_SOURCE_DIR}/shaders_viper")