SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "CMAKE install path" FORCE)
SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_CLEAN_FILES ${CMAKE_BINARY_DIR}/dist)

SET(VEDATensors_INSTALL_PATH "veda/tensors")

INSTALL(FILES ${CMAKE_SOURCE_DIR}/README.md DESTINATION ${VEDATensors_INSTALL_PATH})
ADD_PYTHON_WHEEL(dist ${CMAKE_CURRENT_LIST_DIR}/veda-tensors.json)