ADD_EXECUTABLE			(tungl_ctest ${CMAKE_CURRENT_LIST_DIR}/test.c)
TARGET_LINK_LIBRARIES	(tungl_ctest tungl)
SET_TARGET_PROPERTIES	(tungl_ctest PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN/../lib64")

ADD_EXECUTABLE			(tungl_cpptest ${CMAKE_CURRENT_LIST_DIR}/test.cpp)
TARGET_LINK_LIBRARIES	(tungl_cpptest tungl)
SET_TARGET_PROPERTIES	(tungl_cpptest PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN/../lib64")

INSTALL(TARGETS tungl_ctest tungl_cpptest DESTINATION tests)