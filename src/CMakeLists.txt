INCLUDE_DIRECTORIES(${VEDA_INCLUDES})
#OPTION(BUILD_SHARED "Builds Tungl as shared library." ON)
#IF(${BUILD_SHARED})
#	ADD_LIBRARY(tungl SHARED ${CMAKE_CURRENT_LIST_DIR}/tungl.cpp)
#ELSE()
#	ADD_LIBRARY(tungl STATIC ${CMAKE_CURRENT_LIST_DIR}/tungl.cpp)
#ENDIF()
#
#TARGET_COMPILE_OPTIONS(tungl PRIVATE
#	-Wuninitialized
#	-Wunknown-pragmas
#	-Wunused-function
#	-Wunused-label
#	-Wunused-value
#	-Wunused-variable
#	-Wno-return-local-addr)
#SET_TARGET_PROPERTIES(tungl
#	PROPERTIES
#	CXX_EXTENSIONS	NO
#	SUFFIX			".so.${TUNGL_VERSION_API}"
#	LINK_FLAGS		"-Wl,--no-as-needed -Wl,-z,defs"
#)
#
#INSTALL(TARGETS tungl LIBRARY DESTINATION lib64)