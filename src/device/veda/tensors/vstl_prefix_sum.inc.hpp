// Taken from: https://github.com/takuya-araki/vstl/blob/master/src/vstl/seq/core/prefix_sum.hpp
/**
	BSD 2-Clause License

	Copyright (c) 2020, NEC Corporation
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	* Redistributions of source code must retain the above copyright notice, this
	list of conditions and the following disclaimer.

	* Redistributions in binary form must reproduce the above copyright notice,
	this list of conditions and the following disclaimer in the documentation
	and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstddef>

#define PREFIX_SUM_VLEN_MIN 32

namespace vstl {
namespace seq {
//------------------------------------------------------------------------------
template<typename T>
static inline PREFIX_SUM_NAME(const T* valp, T* outp, size_t size) {
	if(size < PREFIX_SUM_VLEN * PREFIX_SUM_VLEN_MIN) {
		T current_val = 0;
		_Pragma(PREFIX_SUM_VECTOR)
		for(size_t j = 0; j < size; j++) {
			auto loaded_v = valp[j];
			auto next_val = loaded_v + current_val;
			outp[j] = next_val;
			current_val = next_val;
		}
		return;
  	}
	
	size_t each = size / PREFIX_SUM_VLEN;
	if(each % 2 == 0 && each > 1)
		each--;

	size_t rest = size - each * PREFIX_SUM_VLEN;

	T current_val[PREFIX_SUM_VLEN];
	_Pragma(PREFIX_SUM_VREG)
	_Pragma(PREFIX_SUM_VECTOR)
	for(int i = 0; i < PREFIX_SUM_VLEN; i++)
		current_val[i] = 0;

	for(size_t j = 0; j < each; j++) {
		_Pragma(PREFIX_SUM_VECTOR)
		for(size_t i = 0; i < PREFIX_SUM_VLEN; i++) {
			auto loaded_v = valp[j + each * i];
			auto next_val = loaded_v + current_val[i];
			outp[j + each * i] = next_val;
			current_val[i] = next_val;
		}
	}

	size_t rest_idx_start = each * PREFIX_SUM_VLEN;
	T current_val_rest = 0;
	for(size_t j = 0; j < rest; j++) {
		auto loaded_v = valp[j + rest_idx_start];
		auto next_val = loaded_v + current_val_rest;
		outp[j + rest_idx_start] = next_val;
		current_val_rest = next_val;
	}

	_Pragma(PREFIX_SUM_VECTOR)
	for(size_t i = 1; i < PREFIX_SUM_VLEN; i++) {
		T to_add = outp[each * i - 1];
		for(size_t j = each * i; j < each * (i+1); j++)
			outp[j] += to_add;
	}

	T to_add = outp[each * PREFIX_SUM_VLEN - 1];
	_Pragma(PREFIX_SUM_VECTOR)
	for(size_t j = each * PREFIX_SUM_VLEN; j < each * PREFIX_SUM_VLEN + rest; j++)
		outp[j] += to_add;
	
	return;
}

//------------------------------------------------------------------------------
}
}

#undef PREFIX_SUM_VLEN_MIN
