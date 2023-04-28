#include "internal.h"

//------------------------------------------------------------------------------
VEDA_TENSORS_API const char* veda_tensors_get_unary(const VEDATensors_unary_op op) {
	switch(op) {
		case VEDA_TENSORS_UNARY_ABS:		return "VEDA_TENSORS_UNARY_ABS";
		case VEDA_TENSORS_UNARY_ADD:		return "VEDA_TENSORS_UNARY_ADD";
		case VEDA_TENSORS_UNARY_ADDCDIV:	return "VEDA_TENSORS_UNARY_ADDCDIV";
		case VEDA_TENSORS_UNARY_ADDCMUL:	return "VEDA_TENSORS_UNARY_ADDCMUL";
		case VEDA_TENSORS_UNARY_CEIL:		return "VEDA_TENSORS_UNARY_CEIL";
		case VEDA_TENSORS_UNARY_CLAMP:		return "VEDA_TENSORS_UNARY_CLAMP";
		case VEDA_TENSORS_UNARY_COS:		return "VEDA_TENSORS_UNARY_COS";
		case VEDA_TENSORS_UNARY_DIV:		return "VEDA_TENSORS_UNARY_DIV";
		case VEDA_TENSORS_UNARY_EXP:		return "VEDA_TENSORS_UNARY_EXP";
		case VEDA_TENSORS_UNARY_FLOOR:		return "VEDA_TENSORS_UNARY_FLOOR";
		case VEDA_TENSORS_UNARY_IMAG:		return "VEDA_TENSORS_UNARY_IMAG";
		case VEDA_TENSORS_UNARY_ISFINITE:	return "VEDA_TENSORS_UNARY_ISFINITE";
		case VEDA_TENSORS_UNARY_ISINF:		return "VEDA_TENSORS_UNARY_ISINF";
		case VEDA_TENSORS_UNARY_ISNAN:		return "VEDA_TENSORS_UNARY_ISNAN";
		case VEDA_TENSORS_UNARY_LOG:		return "VEDA_TENSORS_UNARY_LOG";
		case VEDA_TENSORS_UNARY_LOG1P:		return "VEDA_TENSORS_UNARY_LOG1P";
		case VEDA_TENSORS_UNARY_MAX:		return "VEDA_TENSORS_UNARY_MAX";
		case VEDA_TENSORS_UNARY_MIN:		return "VEDA_TENSORS_UNARY_MIN";
		case VEDA_TENSORS_UNARY_MUL:		return "VEDA_TENSORS_UNARY_MUL";
		case VEDA_TENSORS_UNARY_NEG:		return "VEDA_TENSORS_UNARY_NEG";
		case VEDA_TENSORS_UNARY_POW:		return "VEDA_TENSORS_UNARY_POW";
		case VEDA_TENSORS_UNARY_REAL:		return "VEDA_TENSORS_UNARY_REAL";
		case VEDA_TENSORS_UNARY_RECIPROCAL:	return "VEDA_TENSORS_UNARY_RECIPROCAL";
		case VEDA_TENSORS_UNARY_RSQRT:		return "VEDA_TENSORS_UNARY_RSQRT";
		case VEDA_TENSORS_UNARY_SIN:		return "VEDA_TENSORS_UNARY_SIN";
		case VEDA_TENSORS_UNARY_SQRT:		return "VEDA_TENSORS_UNARY_SQRT";
		case VEDA_TENSORS_UNARY_SUB:		return "VEDA_TENSORS_UNARY_SUB";
		case VEDA_TENSORS_UNARY_TAN:		return "VEDA_TENSORS_UNARY_TAN";
	}
	return "VEDA_TENSORS_UNARY_UNKNOWN";
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API const char* veda_tensors_get_softmax(const VEDATensors_softmax_op op) {
	switch(op) {
		case VEDA_TENSORS_SOFTMAX_SOFTMAX:		return "VEDA_TENSORS_SOFTMAX_SOFTMAX";
		case VEDA_TENSORS_SOFTMAX_LOGSOFTMAX:	return "VEDA_TENSORS_SOFTMAX_LOGSOFTMAX";
	}
	return "VEDA_TENSORS_SOFTMAX_UNKNOWN";
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API const char* veda_tensors_get_reduce(const VEDATensors_reduce_op op) {
	switch(op) {
		case VEDA_TENSORS_REDUCE_ALL:	return "VEDA_TENSORS_REDUCE_ALL";
		case VEDA_TENSORS_REDUCE_ANY:	return "VEDA_TENSORS_REDUCE_ANY";
		case VEDA_TENSORS_REDUCE_L0:	return "VEDA_TENSORS_REDUCE_L0";
		case VEDA_TENSORS_REDUCE_L1:	return "VEDA_TENSORS_REDUCE_L1";
		case VEDA_TENSORS_REDUCE_L2:	return "VEDA_TENSORS_REDUCE_L2";
		case VEDA_TENSORS_REDUCE_MAX:	return "VEDA_TENSORS_REDUCE_MAX";
		case VEDA_TENSORS_REDUCE_MEAN:	return "VEDA_TENSORS_REDUCE_MEAN";
		case VEDA_TENSORS_REDUCE_MIN:	return "VEDA_TENSORS_REDUCE_MIN";
		case VEDA_TENSORS_REDUCE_SUM:	return "VEDA_TENSORS_REDUCE_SUM";
	}
	return "VEDA_TENSORS_REDUCE_UNKNOWN";
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API const char* veda_tensors_get_dtype(const VEDATensors_dtype dtype) {
	switch(dtype) {
		case VEDA_TENSORS_DTYPE_F32:		return "VEDA_TENSORS_DTYPE_F32";
		case VEDA_TENSORS_DTYPE_F32_F32:	return "VEDA_TENSORS_DTYPE_F32_F32";
		case VEDA_TENSORS_DTYPE_F64:		return "VEDA_TENSORS_DTYPE_F64";
		case VEDA_TENSORS_DTYPE_F64_F64:	return "VEDA_TENSORS_DTYPE_F64_F64";
		case VEDA_TENSORS_DTYPE_S16:		return "VEDA_TENSORS_DTYPE_S16";
		case VEDA_TENSORS_DTYPE_S32:		return "VEDA_TENSORS_DTYPE_S32";
		case VEDA_TENSORS_DTYPE_S64:		return "VEDA_TENSORS_DTYPE_S64";
		case VEDA_TENSORS_DTYPE_S8:			return "VEDA_TENSORS_DTYPE_S8";
		case VEDA_TENSORS_DTYPE_U16:		return "VEDA_TENSORS_DTYPE_U16";
		case VEDA_TENSORS_DTYPE_U32:		return "VEDA_TENSORS_DTYPE_U32";
		case VEDA_TENSORS_DTYPE_U64:		return "VEDA_TENSORS_DTYPE_U64";
		case VEDA_TENSORS_DTYPE_U8:			return "VEDA_TENSORS_DTYPE_U8";
	}
	return "VEDA_TENSORS_DTYPE_UNKNOWN";
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API int veda_tensors_dtype_bytes(const VEDATensors_dtype op) {
	switch(op) {
		case VEDA_TENSORS_DTYPE_F32:		return 4;
		case VEDA_TENSORS_DTYPE_F32_F32:	return 8;
		case VEDA_TENSORS_DTYPE_F64:		return 8;
		case VEDA_TENSORS_DTYPE_F64_F64:	return 16;
		case VEDA_TENSORS_DTYPE_S16:		return 2;
		case VEDA_TENSORS_DTYPE_S32:		return 4;
		case VEDA_TENSORS_DTYPE_S64:		return 8;
		case VEDA_TENSORS_DTYPE_S8:			return 1;
		case VEDA_TENSORS_DTYPE_U16:		return 2;
		case VEDA_TENSORS_DTYPE_U32:		return 4;
		case VEDA_TENSORS_DTYPE_U64:		return 8;
		case VEDA_TENSORS_DTYPE_U8:			return 1;
	}
	THROW("Unknown VEDATensors_dtype!");
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API const char* veda_tensors_get_binary(const VEDATensors_binary_op op) {
	switch(op) {
		case VEDA_TENSORS_BINARY_AND:	return "VEDA_TENSORS_BINARY_AND";
		case VEDA_TENSORS_BINARY_EQ:	return "VEDA_TENSORS_BINARY_EQ";
		case VEDA_TENSORS_BINARY_GE:	return "VEDA_TENSORS_BINARY_GE";
		case VEDA_TENSORS_BINARY_GT:	return "VEDA_TENSORS_BINARY_GT";
		case VEDA_TENSORS_BINARY_LE:	return "VEDA_TENSORS_BINARY_LE";
		case VEDA_TENSORS_BINARY_LT:	return "VEDA_TENSORS_BINARY_LT";
		case VEDA_TENSORS_BINARY_NE:	return "VEDA_TENSORS_BINARY_NE";
		case VEDA_TENSORS_BINARY_OR:	return "VEDA_TENSORS_BINARY_OR";
		case VEDA_TENSORS_BINARY_XOR:	return "VEDA_TENSORS_BINARY_XOR";
	}
	return "VEDA_TENSORS_BINARY_UNKNOWN";
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API const char* veda_tensors_get_bitwise(const VEDATensors_bitwise_op op) {
	switch(op) {
		case VEDA_TENSORS_BITWISE_AND:	return "VEDA_TENSORS_BITWISE_AND";
		case VEDA_TENSORS_BITWISE_OR:	return "VEDA_TENSORS_BITWISE_OR";
		case VEDA_TENSORS_BITWISE_XOR:	return "VEDA_TENSORS_BITWISE_XOR";
	}
	return "VEDA_TENSORS_BITWISE_UNKNOWN";
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API const char* veda_tensors_get_kernel(const VEDATensors_kernel kernel) {
	switch(kernel) {
		case VEDA_TENSORS_KERNEL_ADADELTA:				return "VEDA_TENSORS_KERNEL_ADADELTA";
		case VEDA_TENSORS_KERNEL_ADAGRAD:				return "VEDA_TENSORS_KERNEL_ADAGRAD";
		case VEDA_TENSORS_KERNEL_ADAM:					return "VEDA_TENSORS_KERNEL_ADAM";
		case VEDA_TENSORS_KERNEL_ADAMAX:				return "VEDA_TENSORS_KERNEL_ADAMAX";
		case VEDA_TENSORS_KERNEL_BINARY:				return "VEDA_TENSORS_KERNEL_BINARY";
		case VEDA_TENSORS_KERNEL_BINARY_S:				return "VEDA_TENSORS_KERNEL_BINARY_S";
		case VEDA_TENSORS_KERNEL_BITWISE:				return "VEDA_TENSORS_KERNEL_BITWISE";
		case VEDA_TENSORS_KERNEL_CAT:					return "VEDA_TENSORS_KERNEL_CAT";
		case VEDA_TENSORS_KERNEL_CONVERT:				return "VEDA_TENSORS_KERNEL_CONVERT";
		case VEDA_TENSORS_KERNEL_COPY:					return "VEDA_TENSORS_KERNEL_COPY";
		case VEDA_TENSORS_KERNEL_COUNT:					return "VEDA_TENSORS_KERNEL_COUNT";
		case VEDA_TENSORS_KERNEL_MASKED_FILL:			return "VEDA_TENSORS_KERNEL_MASKED_FILL";
		case VEDA_TENSORS_KERNEL_MASKED_SCATTER:		return "VEDA_TENSORS_KERNEL_MASKED_SCATTER";
		case VEDA_TENSORS_KERNEL_MASKED_SELECT:			return "VEDA_TENSORS_KERNEL_MASKED_SELECT";
		case VEDA_TENSORS_KERNEL_PREFIX_SUM:			return "VEDA_TENSORS_KERNEL_PREFIX_SUM";
		case VEDA_TENSORS_KERNEL_PRINT:					return "VEDA_TENSORS_KERNEL_PRINT";
		case VEDA_TENSORS_KERNEL_REDUCE:				return "VEDA_TENSORS_KERNEL_REDUCE";
		case VEDA_TENSORS_KERNEL_REDUCE_DIM:			return "VEDA_TENSORS_KERNEL_REDUCE_DIM";
		case VEDA_TENSORS_KERNEL_SELECT:				return "VEDA_TENSORS_KERNEL_SELECT";
		case VEDA_TENSORS_KERNEL_SOFTMAX:				return "VEDA_TENSORS_KERNEL_SOFTMAX";
		case VEDA_TENSORS_KERNEL_TRANSPOSE:				return "VEDA_TENSORS_KERNEL_TRANSPOSE";
		case VEDA_TENSORS_KERNEL_UNARY_B:				return "VEDA_TENSORS_KERNEL_UNARY_B";
		case VEDA_TENSORS_KERNEL_UNARY_C:				return "VEDA_TENSORS_KERNEL_UNARY_C";
		case VEDA_TENSORS_KERNEL_UNARY_T:				return "VEDA_TENSORS_KERNEL_UNARY_T";
		case VEDA_TENSORS_KERNEL_UNARY_TS:				return "VEDA_TENSORS_KERNEL_UNARY_TS";
		case VEDA_TENSORS_KERNEL_UNARY_TSS:				return "VEDA_TENSORS_KERNEL_UNARY_TSS";
		case VEDA_TENSORS_KERNEL_UNARY_TT:				return "VEDA_TENSORS_KERNEL_UNARY_TT";
		case VEDA_TENSORS_KERNEL_UNARY_TTS:				return "VEDA_TENSORS_KERNEL_UNARY_TTS";
		case VEDA_TENSORS_KERNEL_UNARY_TTT:				return "VEDA_TENSORS_KERNEL_UNARY_TTT";
		case VEDA_TENSORS_KERNEL_UNARY_TTTS:			return "VEDA_TENSORS_KERNEL_UNARY_TTTS";
		case VEDA_TENSORS_KERNEL_WHERE:					return "VEDA_TENSORS_KERNEL_WHERE";
	}
	return "VEDA_TENSORS_KERNEL_UNKNOWN";
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API const char* veda_tensors_get_kernel_func(const VEDATensors_kernel kernel) {
	switch(kernel) {
		case VEDA_TENSORS_KERNEL_ADADELTA:			return "veda_tensors_adadelta";
		case VEDA_TENSORS_KERNEL_ADAGRAD:			return "veda_tensors_adagrad";
		case VEDA_TENSORS_KERNEL_ADAM:				return "veda_tensors_adam";
		case VEDA_TENSORS_KERNEL_ADAMAX:			return "veda_tensors_adamax";
		case VEDA_TENSORS_KERNEL_BINARY:			return "veda_tensors_binary";
		case VEDA_TENSORS_KERNEL_BINARY_S:			return "veda_tensors_binary_s";
		case VEDA_TENSORS_KERNEL_BITWISE:			return "veda_tensors_bitwise";
		case VEDA_TENSORS_KERNEL_CAT:				return "veda_tensors_cat";
		case VEDA_TENSORS_KERNEL_CONVERT:			return "veda_tensors_convert";
		case VEDA_TENSORS_KERNEL_COPY:				return "veda_tensors_copy";
		case VEDA_TENSORS_KERNEL_COUNT:				return "veda_tensors_count";
		case VEDA_TENSORS_KERNEL_MASKED_FILL:		return "veda_tensors_masked_fill";
		case VEDA_TENSORS_KERNEL_MASKED_SCATTER:	return "veda_tensors_masked_scatter";
		case VEDA_TENSORS_KERNEL_MASKED_SELECT:		return "veda_tensors_masked_select";
		case VEDA_TENSORS_KERNEL_PREFIX_SUM:		return "veda_tensors_prefix_sum";
		case VEDA_TENSORS_KERNEL_PRINT:				return "veda_tensors_print";
		case VEDA_TENSORS_KERNEL_REDUCE:			return "veda_tensors_reduce";
		case VEDA_TENSORS_KERNEL_REDUCE_DIM:		return "veda_tensors_reduce_dim";
		case VEDA_TENSORS_KERNEL_SELECT:			return "veda_tensors_select";
		case VEDA_TENSORS_KERNEL_SOFTMAX:			return "veda_tensors_softmax";
		case VEDA_TENSORS_KERNEL_TRANSPOSE:			return "veda_tensors_transpose";
		case VEDA_TENSORS_KERNEL_UNARY_B:			return "veda_tensors_unary_b";
		case VEDA_TENSORS_KERNEL_UNARY_C:			return "veda_tensors_unary_c";
		case VEDA_TENSORS_KERNEL_UNARY_T:			return "veda_tensors_unary_t";
		case VEDA_TENSORS_KERNEL_UNARY_TS:			return "veda_tensors_unary_ts";
		case VEDA_TENSORS_KERNEL_UNARY_TSS:			return "veda_tensors_unary_tss";
		case VEDA_TENSORS_KERNEL_UNARY_TT:			return "veda_tensors_unary_tt";
		case VEDA_TENSORS_KERNEL_UNARY_TTS:			return "veda_tensors_unary_tts";
		case VEDA_TENSORS_KERNEL_UNARY_TTT:			return "veda_tensors_unary_ttt";
		case VEDA_TENSORS_KERNEL_UNARY_TTTS:		return "veda_tensors_unary_ttts";
		case VEDA_TENSORS_KERNEL_WHERE:				return "veda_tensors_where";
	}
	THROW("Unknown VEDA_TENSOR_KERNEL %i", kernel);
}

//------------------------------------------------------------------------------
