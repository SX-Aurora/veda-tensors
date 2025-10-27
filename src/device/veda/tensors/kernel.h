// Copyright 2022-2025 NEC Laboratories Europe

#pragma once 

#define KERNEL(TYPE, ...)\
	switch(TYPE) {\
		__VA_ARGS__\
		default:\
			FAIL();\
	}

#define KERNEL_UINT(FUNC, ...)\
	case VEDA_TENSORS_DTYPE_U8:		FUNC<uint8_t>	(__VA_ARGS__);	break;\
	case VEDA_TENSORS_DTYPE_U16:	FUNC<uint16_t>	(__VA_ARGS__);	break;\
	case VEDA_TENSORS_DTYPE_U32:	FUNC<uint32_t>	(__VA_ARGS__);	break;\
	case VEDA_TENSORS_DTYPE_U64:	FUNC<uint64_t>	(__VA_ARGS__);	break;

#define KERNEL_FLOAT_(FUNC, ...)\
	case VEDA_TENSORS_DTYPE_F32:	FUNC<float>		(__VA_ARGS__);	break;\
	case VEDA_TENSORS_DTYPE_F64:	FUNC<double>	(__VA_ARGS__);	break;

#define KERNEL_SINT(FUNC, ...)\
	case VEDA_TENSORS_DTYPE_S8:		FUNC<int8_t>	(__VA_ARGS__); break;\
	case VEDA_TENSORS_DTYPE_S16:	FUNC<int16_t>	(__VA_ARGS__); break;\
	case VEDA_TENSORS_DTYPE_S32:	FUNC<int32_t>	(__VA_ARGS__); break;\
	case VEDA_TENSORS_DTYPE_S64:	FUNC<int64_t>	(__VA_ARGS__); break;

#define KERNEL_STRUCTS(FUNC, ...)\
	case VEDA_TENSORS_DTYPE_F64_F64:	FUNC<double,	double_double>	(__VA_ARGS__); break;\
	case VEDA_TENSORS_DTYPE_F32_F32:	FUNC<float,		float_float>	(__VA_ARGS__); break;

#define KERNEL_FLOAT(TYPE, FUNC, ...)\
	KERNEL(TYPE,\
		KERNEL_FLOAT_(FUNC, __VA_ARGS__)\
	)

#define KERNEL_PRIMITIVE(TYPE, FUNC, ...)\
	KERNEL(TYPE,\
		KERNEL_SINT(FUNC, __VA_ARGS__)\
		KERNEL_UINT(FUNC, __VA_ARGS__)\
		KERNEL_FLOAT_(FUNC, __VA_ARGS__)\
	)

#define KERNEL_INTEGER(TYPE, FUNC, ...)\
	KERNEL(TYPE,\
		KERNEL_SINT(FUNC, __VA_ARGS__)\
		KERNEL_UINT(FUNC, __VA_ARGS__)\
	)

#define KERNEL_ALL(TYPE, FUNC, ...)\
	KERNEL(TYPE,\
		KERNEL_SINT(FUNC, __VA_ARGS__)\
		KERNEL_UINT(FUNC, __VA_ARGS__)\
		KERNEL_FLOAT_(FUNC, __VA_ARGS__)\
		KERNEL_STRUCTS(FUNC, __VA_ARGS__)\
	)

#define ATTR_PTR(P)		VEDAdeviceptr _##P
#define ATTR_SCALAR(S)	const uint64_t S##_x, const uint64_t S##_y
#define DEREF_PTR(P)	auto P = VEDAptr<void>(_##P).ptr()
#define DEREF_SCALAR(S)	const uint64_t S[] = {S##_x, S##_y}
