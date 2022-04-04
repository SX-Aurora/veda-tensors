ILLYRIAN_INCLUDE_DIRECTORIES(TARGETS veda-tensors PATHS ${CMAKE_CURRENT_LIST_DIR})

SET(SRC)
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/veda/tensors/CMakeLists.txt)

ADD_LIBRARY				(veda-tensors_device SHARED ${SRC})
TARGET_COMPILE_OPTIONS	(veda-tensors_device PRIVATE -fopenmp)
SET_TARGET_PROPERTIES 	(veda-tensors_device PROPERTIES LINK_FLAGS "-fopenmp -Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/veda-tensors.map")
SET_TARGET_PROPERTIES	(veda-tensors_device PROPERTIES OUTPUT_NAME "veda-tensors")
TARGET_LINK_LIBRARIES	(veda-tensors_device ${VEDA_DEVICE_LIBRARY})
INSTALL					(TARGETS veda-tensors_device LIBRARY DESTINATION ${VEDATensors_INSTALL_PATH}/libve)