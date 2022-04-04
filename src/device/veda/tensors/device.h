// Copyright 2022 NEC Laboratories Europe

#pragma once 

#include <stdint.h>
#include <veda_device.h>
#include <veda/tensors/enums.h>

__global__	uint64_t	veda_tensors_count			(VEDAdeviceptr ptr, const uint64_t elements, const int bytes);
__global__	void		veda_tensors_binary			(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t co, const size_t cx, const size_t cy, const int op, const int type);
__global__	void		veda_tensors_binary			(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t co, const size_t cx, const size_t cy, const int op, const int type);
__global__	void		veda_tensors_bitwise		(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t co, const size_t cx, const size_t cy, const int op, const int bytes);
__global__	void		veda_tensors_cat			(const int cnt, VEDAdeviceptr output, const VEDAdeviceptr inputs, const VEDAdeviceptr offsets, const VEDAdeviceptr center, const size_t left, const size_t right, const int bytes);
__global__	void		veda_tensors_convert		(VEDAdeviceptr dst, VEDAdeviceptr src, const uint64_t elements, const int dstType, const int srcType);
__global__	void		veda_tensors_copy			(VEDAdeviceptr o, VEDAdeviceptr x, const size_t cnt, const int bytes);
__global__	void		veda_tensors_masked_fill	(VEDAdeviceptr out, VEDAdeviceptr value, VEDAdeviceptr mask, const size_t elements, const size_t otherElements, const int bytes);
__global__	void		veda_tensors_masked_scatter	(VEDAdeviceptr out, VEDAdeviceptr value, VEDAdeviceptr mask, const size_t elements, const int bits);
__global__	void		veda_tensors_masked_select	(VEDAdeviceptr out, VEDAdeviceptr in, VEDAdeviceptr mask, const size_t outElements, const size_t elements, const int bytes);
__global__	void		veda_tensors_prefix_sum		(VEDAdeviceptr out, VEDAdeviceptr carry, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const int type, const int inclusive);
__global__	void		veda_tensors_print			(VEDAdeviceptr ptr, const uint64_t elements, const int type);
__global__	void		veda_tensors_reduce			(VEDAdeviceptr out, VEDAdeviceptr in, const uint64_t elements, const int op, const int type);
__global__	void		veda_tensors_reduce_dim		(VEDAdeviceptr values, VEDAdeviceptr indices, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const int op, const int type);
__global__	void		veda_tensors_select			(VEDAdeviceptr out, VEDAdeviceptr in, const size_t ocnt, const size_t ostride, const size_t icnt, const size_t offset, const int bytes);
__global__	void		veda_tensors_transpose		(VEDAdeviceptr dst, VEDAdeviceptr src, const uint64_t sizeA, const uint64_t sizeB, const int bytes);
__global__	void		veda_tensors_unary_b		(VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const int op, const int type);
__global__	void		veda_tensors_unary_c		(VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const int op, const int type);
__global__	void		veda_tensors_unary_t		(VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const int op, const int type);
__global__	void		veda_tensors_unary_tt		(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t co, const size_t cx, const size_t cy, const int op, const int type);
__global__	void		veda_tensors_unary_tts		(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const uint64_t alpha_x, const uint64_t alpha_y, const size_t co, const size_t cx, const size_t cy, const int op, const int type);
__global__	void		veda_tensors_unary_ttts		(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, VEDAdeviceptr z, const uint64_t alpha_x, const uint64_t alpha_y, const size_t oc, const size_t ox, const size_t oy, const size_t oz, const int op, const int type);
