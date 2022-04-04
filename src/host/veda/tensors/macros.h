// Copyright 2022 NEC Laboratories Europe

#pragma once 

//------------------------------------------------------------------------------
#define VEDA_TENSORS_MAX_DIMS 8
#define VEDA_TENSORS_UNLIKELY(...)		__builtin_expect((__VA_ARGS__),0)
#define VEDA_TENSORS_LIKELY(...)		__builtin_expect((__VA_ARGS__),1)
#define VEDA_TENSORS_THROW(COND, ...)	if(VEDA_TENSORS_UNLIKELY(COND)) {	L_ERROR(__VA_ARGS__); assert(false);	throw VEDA_ERROR_INVALID_ARGS;	}

//------------------------------------------------------------------------------
