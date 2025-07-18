CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

PROJECT(VEDATensors VERSION 0.1.8 LANGUAGES C CXX)

FIND_PACKAGE(Illyrian 0.3.0 REQUIRED)
ILLYRIAN_PROJECT(
	C_STANDARD		17
	CXX_STANDARD	17
	INSTALL_PREFIX "/usr/local/ve/"
)

ILLYRIAN_FIND_PACKAGE(VEDA  VERSION 2.1.0 REQUIRED PYTHON "veda/cmake")
ILLYRIAN_FIND_PACKAGE(Tungl VERSION 0.1.1 REQUIRED PYTHON "tungl/cmake")

SET(CMAKE_MODULE_PATH ${VEDA_DIR}/cmake)
ENABLE_LANGUAGE(VEDA_CXX)

ILLYRIAN_OPTIONS(VEDATensors_BUILD_TYPE SHARED STATIC)
ILLYRIAN_OPTIONS(VEDATensors_DIST_TYPE  LOCAL  PYTHON)
INCLUDE(dist/CMakeLists.txt)

ILLYRIAN_INSTALL_FIND_BEGIN		(VEDATensors PYTHON_NAME veda-tensors FILE "include/veda/tensors/api.h" PATHS "/usr/local/ve/veda-tensors/")
ILLYRIAN_INSTALL_FIND_REQUIRED	(LIBRARY VEDATensors LIBRARY		"libveda-tensors.so" "libveda-tensors.a"	PATHS "\${VEDATensors_DIR}/lib64")
ILLYRIAN_INSTALL_FIND_REQUIRED	(FILE VEDATensors DEVICE_LIBRARY	"libveda-tensors.vso"						PATHS "\${VEDATensors_DIR}/libve")
ILLYRIAN_INSTALL_FIND_SET		(VEDATensors INCLUDE_DIRS "\${VEDATensors_DIR}/include" CACHE STRING "VEDATensors include dir")
ILLYRIAN_INSTALL_FIND_END		(VEDATensors
	VERSION_FILE "\${VEDATensors_INCLUDE_DIRS}/veda/tensors/version.h"
	VERSION_REGEX "\\\"([0-9\.]+)\\\""
	DESTINATION ${VEDATensors_INSTALL_PATH}/cmake
)

## Build VEDA-Tensors ----------------------------------------------------------
INCLUDE_DIRECTORIES(${Tungl_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${VEDA_INCLUDE_DIRS})

ADD_SUBDIRECTORY(src/host)
ADD_SUBDIRECTORY(src/device)

OPTION(WITH_TESTS "Enable tests" OFF)
IF(WITH_TESTS)
	# TODO: ADD_SUBDIRECTORY(src/tests)
ENDIF()

INSTALL(FILES ${CMAKE_CURRENT_LIST_DIR}/LICENSE DESTINATION ${VEDATensors_INSTALL_PATH})