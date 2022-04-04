SET(SRC
	${CMAKE_CURRENT_LIST_DIR}/binary.vcpp
	${CMAKE_CURRENT_LIST_DIR}/bitwise.vcpp
	${CMAKE_CURRENT_LIST_DIR}/cat.vcpp
	${CMAKE_CURRENT_LIST_DIR}/convert.vcpp
	${CMAKE_CURRENT_LIST_DIR}/copy.vcpp
	${CMAKE_CURRENT_LIST_DIR}/count.vcpp
	${CMAKE_CURRENT_LIST_DIR}/transpose.vcpp
	${CMAKE_CURRENT_LIST_DIR}/masked_fill.vcpp
	${CMAKE_CURRENT_LIST_DIR}/masked_scatter.vcpp
	${CMAKE_CURRENT_LIST_DIR}/masked_select.vcpp
	${CMAKE_CURRENT_LIST_DIR}/prefixsum.vcpp
	${CMAKE_CURRENT_LIST_DIR}/print.vcpp
	${CMAKE_CURRENT_LIST_DIR}/reduce.vcpp
	${CMAKE_CURRENT_LIST_DIR}/select.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_b.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_c.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_t.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_tt.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_tts.vcpp
	${CMAKE_CURRENT_LIST_DIR}/unary_ttts.vcpp
)

INSTALL(FILES ${CMAKE_CURRENT_LIST_DIR}/device.h DESTINATION ${VEDATensors_INSTALL_PATH}/include/veda/tensors)