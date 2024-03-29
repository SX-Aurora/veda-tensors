// Copyright 2022 NEC Laboratories Europe

#pragma once 

//------------------------------------------------------------------------------
typedef enum {
	VEDA_TENSORS_UNARY_UNKNOWN		= -1,
	VEDA_TENSORS_UNARY_ABS			= 0,
	VEDA_TENSORS_UNARY_SQRT			= 1,
	VEDA_TENSORS_UNARY_RSQRT		= 2,
	VEDA_TENSORS_UNARY_SIN			= 3,
	VEDA_TENSORS_UNARY_COS			= 4,
	VEDA_TENSORS_UNARY_TAN			= 5,
	VEDA_TENSORS_UNARY_EXP			= 6,
	VEDA_TENSORS_UNARY_LOG			= 7,
	VEDA_TENSORS_UNARY_MIN			= 8,
	VEDA_TENSORS_UNARY_MAX			= 9,
	VEDA_TENSORS_UNARY_CEIL			= 10,
	VEDA_TENSORS_UNARY_FLOOR		= 11,
	VEDA_TENSORS_UNARY_ADD			= 12,
	VEDA_TENSORS_UNARY_SUB			= 13,
	VEDA_TENSORS_UNARY_MUL			= 14,
	VEDA_TENSORS_UNARY_DIV			= 15,
	VEDA_TENSORS_UNARY_RECIPROCAL	= 16,
	VEDA_TENSORS_UNARY_ADDCMUL		= 17,
	VEDA_TENSORS_UNARY_ADDCDIV		= 18,
	VEDA_TENSORS_UNARY_ISFINITE		= 19,
	VEDA_TENSORS_UNARY_ISINF		= 20,
	VEDA_TENSORS_UNARY_ISNAN		= 21,
	VEDA_TENSORS_UNARY_REAL			= 22,
	VEDA_TENSORS_UNARY_IMAG			= 23,
	VEDA_TENSORS_UNARY_NEG			= 24,
	VEDA_TENSORS_UNARY_CLAMP		= 25,
	VEDA_TENSORS_UNARY_POW			= 26,
	VEDA_TENSORS_UNARY_LOG1P		= 27,
	VEDA_TENSORS_UNARY_NOT			= 28,
} VEDATensors_unary_op;

//------------------------------------------------------------------------------
typedef enum {
	VEDA_TENSORS_SOFTMAX_UNKNOWN	= -1,
	VEDA_TENSORS_SOFTMAX_SOFTMAX	= 0,
	VEDA_TENSORS_SOFTMAX_LOGSOFTMAX	= 1
} VEDATensors_softmax_op;

//------------------------------------------------------------------------------
typedef enum {
	VEDA_TENSORS_REDUCE_UNKNOWN	= -1,
	VEDA_TENSORS_REDUCE_MIN		= 0,
	VEDA_TENSORS_REDUCE_MAX		= 1,
	VEDA_TENSORS_REDUCE_MEAN	= 2,
	VEDA_TENSORS_REDUCE_SUM		= 3,
	VEDA_TENSORS_REDUCE_L0		= 4,
	VEDA_TENSORS_REDUCE_L1		= 5,
	VEDA_TENSORS_REDUCE_L2		= 6,
	VEDA_TENSORS_REDUCE_ANY		= 7,
	VEDA_TENSORS_REDUCE_ALL		= 8
} VEDATensors_reduce_op;

//------------------------------------------------------------------------------
typedef enum {
	VEDA_TENSORS_DTYPE_UNKNOWN	= -1,
	VEDA_TENSORS_DTYPE_S8		= 0,
	VEDA_TENSORS_DTYPE_S16		= 1,
	VEDA_TENSORS_DTYPE_S32		= 2,
	VEDA_TENSORS_DTYPE_S64		= 3,
	VEDA_TENSORS_DTYPE_U8		= 4,
	VEDA_TENSORS_DTYPE_U16		= 5,
	VEDA_TENSORS_DTYPE_U32		= 6,
	VEDA_TENSORS_DTYPE_U64		= 7,
	VEDA_TENSORS_DTYPE_F32		= 8,
	VEDA_TENSORS_DTYPE_F64		= 9,
	VEDA_TENSORS_DTYPE_F32_F32	= 10,
	VEDA_TENSORS_DTYPE_F64_F64	= 11
} VEDATensors_dtype;

//------------------------------------------------------------------------------
typedef enum {
	VEDA_TENSORS_BINARY_UNKNOWN	= -1,
	VEDA_TENSORS_BINARY_EQ		= 0,
	VEDA_TENSORS_BINARY_NE		= 1,
	VEDA_TENSORS_BINARY_LT		= 2,
	VEDA_TENSORS_BINARY_LE		= 3,
	VEDA_TENSORS_BINARY_GT		= 4,
	VEDA_TENSORS_BINARY_GE		= 5,
	VEDA_TENSORS_BINARY_AND		= 6,
	VEDA_TENSORS_BINARY_OR		= 7,
	VEDA_TENSORS_BINARY_XOR		= 8
} VEDATensors_binary_op;

//------------------------------------------------------------------------------
typedef enum {
	VEDA_TENSORS_BITWISE_UNKNOWN	= -1,
	VEDA_TENSORS_BITWISE_AND		= 0,
	VEDA_TENSORS_BITWISE_OR			= 1,
	VEDA_TENSORS_BITWISE_XOR		= 2
} VEDATensors_bitwise_op;

//------------------------------------------------------------------------------
typedef enum {
	VEDA_TENSORS_KERNEL_UNKNOWN				= -1,
	VEDA_TENSORS_KERNEL_BINARY				= 0,
	VEDA_TENSORS_KERNEL_BITWISE				= 1,
	VEDA_TENSORS_KERNEL_CAT					= 2,
	VEDA_TENSORS_KERNEL_CONVERT				= 3,
	VEDA_TENSORS_KERNEL_COPY				= 4,
	VEDA_TENSORS_KERNEL_COUNT				= 5,
	VEDA_TENSORS_KERNEL_TRANSPOSE			= 6,
	VEDA_TENSORS_KERNEL_MASKED_FILL			= 7,
	VEDA_TENSORS_KERNEL_MASKED_SCATTER		= 8,
	VEDA_TENSORS_KERNEL_MASKED_SELECT		= 9,
	VEDA_TENSORS_KERNEL_PREFIX_SUM			= 10,
	VEDA_TENSORS_KERNEL_PRINT				= 11,
	VEDA_TENSORS_KERNEL_REDUCE				= 12,
	VEDA_TENSORS_KERNEL_REDUCE_DIM			= 13,
	VEDA_TENSORS_KERNEL_SELECT				= 14,
	VEDA_TENSORS_KERNEL_UNARY_B				= 15,
	VEDA_TENSORS_KERNEL_UNARY_C				= 16,
	VEDA_TENSORS_KERNEL_UNARY_T				= 17,
	VEDA_TENSORS_KERNEL_UNARY_TS			= 18,
	VEDA_TENSORS_KERNEL_UNARY_TT			= 19,
	VEDA_TENSORS_KERNEL_UNARY_TSS			= 20,
	VEDA_TENSORS_KERNEL_UNARY_TTS			= 21,
	VEDA_TENSORS_KERNEL_UNARY_TTT			= 22,
	VEDA_TENSORS_KERNEL_UNARY_TTTS			= 23,
	VEDA_TENSORS_KERNEL_ADADELTA			= 24,
	VEDA_TENSORS_KERNEL_ADAGRAD				= 25,
	VEDA_TENSORS_KERNEL_ADAM				= 26,
	VEDA_TENSORS_KERNEL_ADAMAX				= 27,
	VEDA_TENSORS_KERNEL_WHERE				= 28,
	VEDA_TENSORS_KERNEL_BINARY_S			= 29,
	VEDA_TENSORS_KERNEL_SOFTMAX				= 30,
	VEDA_TENSORS_KERNEL_CNT					= 31
} VEDATensors_kernel;

//------------------------------------------------------------------------------
