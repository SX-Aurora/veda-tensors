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