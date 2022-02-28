// Copyright 2017-2022 NEC Laboratories Europe

#pragma once

template<typename T>
VEDAresult veda_tensors_prefixsum(T* out, T* carry, const T* in, const size_t left, const size_t center, const size_t right, const bool inclusive);

template<typename T>
inline VEDAresult veda_tensors_prefixsum(T* out, const T* in, const size_t left, const size_t center, const size_t right, const bool inclusive) {
	return veda_tensors_prefixsum(out, (T*)0, in, left, center, right, inclusive);
}
