ILLYRIAN_INCLUDE_DIRECTORIES(TARGETS veda-tensors PATHS ${CMAKE_CURRENT_LIST_DIR})

SET(SRC)
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/veda/tensors/CMakeLists.txt)

FUNCTION(ADD_DEVICE_LIBRARY ARCH SUFFIX VEDA_LIBRARY)
	ADD_LIBRARY				(veda-tensors_device-${ARCH} SHARED ${SRC})
	TARGET_COMPILE_OPTIONS	(veda-tensors_device-${ARCH} PRIVATE -fopenmp -march=${ARCH} -finline -finline-functions -finline-max-depth=10)
	SET_TARGET_PROPERTIES 	(veda-tensors_device-${ARCH} PROPERTIES LINK_FLAGS "-march=${ARCH} -fopenmp -Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/veda-tensors.map")
	SET_TARGET_PROPERTIES	(veda-tensors_device-${ARCH} PROPERTIES OUTPUT_NAME "veda-tensors")
	SET_TARGET_PROPERTIES	(veda-tensors_device-${ARCH} PROPERTIES SUFFIX ${SUFFIX})
	TARGET_LINK_LIBRARIES	(veda-tensors_device-${ARCH} ${VEDA_LIBRARY})
	INSTALL					(TARGETS veda-tensors_device-${ARCH} LIBRARY DESTINATION ${VEDATensors_INSTALL_PATH}/libve)
ENDFUNCTION()

ADD_DEVICE_LIBRARY(ve1 ".vso" ${VEDA_DEVICE_LIBRARY})
ADD_DEVICE_LIBRARY(ve3 ".vso3" ${VEDA_DEVICE3_LIBRARY})