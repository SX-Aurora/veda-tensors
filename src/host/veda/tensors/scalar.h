// Copyright 2022 NEC Laboratories Europe

#pragma once 

typedef union {
	int8_t		S8;
	int16_t		S16;
	int32_t		S32;
	int64_t		S64;
	uint8_t		U8;
	uint16_t	U16;
	uint32_t	U32;
	uint64_t	U64;
	float		F32;
	double		F64;
	struct {	float x; float y;	}		F32_F32;
	struct {	double x; double y;	}		F64_F64;
	struct {	uint64_t x; uint64_t y;	}	U64_U64;

} VEDATensors_scalar;

#if __cplusplus
template<typename T> VEDATensors_scalar veda_tensors_scalar(const T value);
template<typename T> VEDATensors_scalar veda_tensors_scalar(const T x, const T y);

template<>	inline	VEDATensors_scalar veda_tensors_scalar(const int8_t value)				{	VEDATensors_scalar scalar; scalar.S8  = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const int16_t value)				{	VEDATensors_scalar scalar; scalar.S16 = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const int32_t value)				{	VEDATensors_scalar scalar; scalar.S32 = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const int64_t value)				{	VEDATensors_scalar scalar; scalar.S64 = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const uint8_t value)				{	VEDATensors_scalar scalar; scalar.U8  = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const uint16_t value)			{	VEDATensors_scalar scalar; scalar.U16 = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const uint32_t value)			{	VEDATensors_scalar scalar; scalar.U32 = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const uint64_t value)			{	VEDATensors_scalar scalar; scalar.U64 = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const float value)				{	VEDATensors_scalar scalar; scalar.F32 = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const double value)				{	VEDATensors_scalar scalar; scalar.F64 = value;	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const float x, const float y)	{	VEDATensors_scalar scalar; scalar.F32_F32 = {x, y};	return scalar;	}
template<>	inline	VEDATensors_scalar veda_tensors_scalar(const double x, const double y)	{	VEDATensors_scalar scalar; scalar.F64_F64 = {x, y};	return scalar;	}
#endif
