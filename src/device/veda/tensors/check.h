// Copyright 2022 NEC Laboratories Europe

#pragma once

#define FVEDA(...)	veda_tensors_check(__VA_ARGS__, __FILE__, __LINE__)

inline void veda_tensors_check(VEDAresult err, const char* file, const int line) {
	if(__builtin_expect((err != VEDA_SUCCESS),0)) {
		const char *name, *string;
		vedaGetErrorName(err, &name);
		vedaGetErrorString(err, &string);
		THROWAT(L_MODULE, file, line, "%s: %s", name, string);
		FAIL();
	}
}
