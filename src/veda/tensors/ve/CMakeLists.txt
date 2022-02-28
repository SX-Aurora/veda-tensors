SET(SET
	${CMAKE_CURRENT_LIST_DIR}/binary.vcpp
	${CMAKE_CURRENT_LIST_DIR}/bitwise.vcpp
	${CMAKE_CURRENT_LIST_DIR}/cat.vcpp
	${CMAKE_CURRENT_LIST_DIR}/check.vcpp
	${CMAKE_CURRENT_LIST_DIR}/convert.vcpp
	${CMAKE_CURRENT_LIST_DIR}/copy.vcpp
	${CMAKE_CURRENT_LIST_DIR}/count.vcpp
	${CMAKE_CURRENT_LIST_DIR}/d2d.vcpp
	${CMAKE_CURRENT_LIST_DIR}/masked_fill.vcpp
	${CMAKE_CURRENT_LIST_DIR}/masked_scatter.vcpp
	${CMAKE_CURRENT_LIST_DIR}/masked_select.vcpp
	${CMAKE_CURRENT_LIST_DIR}/prefixsum.vcpp
	${CMAKE_CURRENT_LIST_DIR}/print.vcpp
	${CMAKE_CURRENT_LIST_DIR}/profiler.vcpp
	${CMAKE_CURRENT_LIST_DIR}/reduce.vcpp
	${CMAKE_CURRENT_LIST_DIR}/select.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_b.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_c.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_t.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_tt.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_tts.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_ttts.vcpp
)

ADD_LIBRARY(veda-tensors-device SHARED ${SRC})
TARGET_LINK_LIBRARIES(veda-tensors-device ${VEDA_DEVICE_LIBRARY})