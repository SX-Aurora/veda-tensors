#include "internal.h"

//------------------------------------------------------------------------------
#define KERNEL(N) handle->kernel[VEDA_TENSORS_KERNEL_##N]
#define GUARD(MSG, ...)\
	if(!handle) return VEDA_ERROR_INVALID_MODULE;\
	L_TRACE("[ve:%i] " MSG, handle->idx, __VA_ARGS__);\
	Guard __guard__(handle)

//------------------------------------------------------------------------------
struct Guard {
	inline Guard(VEDATensors_chandle handle) {
		THROWIF(vedaCtxPushCurrent(handle->ctx) != VEDA_SUCCESS, "Unable to push VEDAcontext");
	}

	inline ~Guard(void) {
		VEDAcontext ctx;
		THROWIF(vedaCtxPopCurrent(&ctx) != VEDA_SUCCESS, "Unable to pop VEDAcontext");
	}
};

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_copy(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t oc, const size_t ox, const VEDATensors_dtype dtype) {
	GUARD("copy(%p, %p, %llu, %llu, %s)", o, x, oc, ox, veda_tensors_get_dtype(dtype));
	auto bytes = veda_tensors_dtype_bytes(dtype);
	if(bytes == 0)	return VEDA_ERROR_NOT_IMPLEMENTED;
	if(ox == 1)		return vedaLaunchKernel(KERNEL(COPY), 0, o, x, oc, bytes);
	if(oc == ox)	return VEDA_ERROR_INVALID_ARGS;
	return vedaMemcpyDtoD(o, x, oc * bytes);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_fill(VEDATensors_chandle handle, VEDAdeviceptr ptr, const VEDATensors_scalar value, const size_t cnt, const VEDATensors_dtype dtype) {
	if(!handle) return VEDA_ERROR_INVALID_MODULE;
	if(tungl_is_active(TUNGL_LEVEL_TRACE)) {
		switch(dtype) {
			#define GEN(T, F, ...) case VEDA_TENSORS_DTYPE_##T: L_TRACE("[ve:%i] fill(%p, " F ", %llu, %s)", handle->idx, ptr, __VA_ARGS__,  cnt, veda_tensors_get_dtype(dtype));	break
			GEN(S8,	 		"%lli",		(int64_t) value.S8);
			GEN(S16,		"%lli",		(int64_t) value.S16);
			GEN(S32,		"%lli",		(int64_t) value.S32);
			GEN(S64,		"%lli",		(int64_t) value.S64);
			GEN(U8,	 		"%llu",		(uint64_t)value.U8);
			GEN(U16,		"%llu",		(uint64_t)value.U16);
			GEN(U32,		"%llu",		(uint64_t)value.U32);
			GEN(U64,		"%llu",		(uint64_t)value.U64);
			GEN(F32,		"%f",		value.F32);
			GEN(F64,		"%f",		value.F64);
			GEN(F32_F32,	"{%f, %f}",	value.F32_F32.x, value.F32_F32.y);
			GEN(F64_F64,	"{%f, %f}",	value.F64_F64.x, value.F64_F64.y);
			#undef GEN
		}
	}
	Guard __guard__(handle);

	switch(veda_tensors_dtype_bytes(dtype)) {
		case 1:		return vedaMemsetD8Async  (ptr, value.U8,  cnt, 0);
		case 2:		return vedaMemsetD16Async (ptr, value.U16, cnt, 0);
		case 4: 	return vedaMemsetD32Async (ptr, value.U32, cnt, 0);
		case 8: 	return vedaMemsetD64Async (ptr, value.U64, cnt, 0);
		case 16:	return vedaMemsetD128Async(ptr, value.U64_U64.x, value.U64_U64.y, cnt, 0);
	}
	return VEDA_ERROR_NOT_IMPLEMENTED;
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_prefix_sum(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr carry, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const VEDATensors_dtype dtype, const int inclusive) {
	GUARD("prefix_sum(%p, %p, %p, %llu, %llu, %llu, %s, %s)", out, carry, in, left, center, right, veda_tensors_get_dtype(dtype), (inclusive ? "inclusive" : "exclusive"));
	return vedaLaunchKernel(KERNEL(PREFIX_SUM), 0, out, carry, in, left, center, right, dtype, inclusive);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_select(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr in, const size_t ocnt, const size_t ostride, const size_t icnt, const size_t offset, const VEDATensors_dtype dtype) {
	GUARD("select(%p, %p, %llu, %llu, %llu, %llu, %s)", out, in, ocnt, ostride, icnt, offset, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(SELECT), 0, out, in, ocnt, ostride, icnt, offset, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_unary_b(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_ISFINITE:
		case VEDA_TENSORS_UNARY_ISINF:
		case VEDA_TENSORS_UNARY_ISNAN:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}
	
	GUARD("unary_b(%p, %p, %llu, %llu, %s, %s)", o, x, co, cx, veda_tensors_get_unary(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_B), 0, o, x, co, cx, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_unary_t(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_ABS:
		case VEDA_TENSORS_UNARY_FLOOR:
		case VEDA_TENSORS_UNARY_CEIL:
		case VEDA_TENSORS_UNARY_RECIPROCAL:
		case VEDA_TENSORS_UNARY_SQRT:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}

	GUARD("unary_t(%p, %p, %llu, %llu, %s, %s)", o, x, co, cx, veda_tensors_get_unary(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_T), 0, o, x, co, cx, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_unary_c(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_ABS:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}

	switch(dtype) {
		case VEDA_TENSORS_DTYPE_F32_F32:
		case VEDA_TENSORS_DTYPE_F64_F64:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}

	GUARD("unary_c(%p, %p, %llu, %llu, %s, %s)", o, x, co, cx, veda_tensors_get_unary(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_C), 0, o, x, co, cx, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_unary_tt(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t co, const size_t cx, const size_t cy, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_MIN:
		case VEDA_TENSORS_UNARY_MAX:
		case VEDA_TENSORS_UNARY_ADD:
		case VEDA_TENSORS_UNARY_SUB:
		case VEDA_TENSORS_UNARY_DIV:
		case VEDA_TENSORS_UNARY_MUL:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}
	
	GUARD("unary_tt(%p, %p, %p, %llu, %llu, %llu, %s, %s)", o, x, y, co, cx, cy, veda_tensors_get_unary(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_TT), 0, o, x, y, co, cx, cy, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_unary_tts(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const VEDATensors_scalar alpha, const size_t co, const size_t cx, const size_t cy, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_ADD:
		case VEDA_TENSORS_UNARY_SUB:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}

	GUARD("unary_tts(%p, %p, %p, %p, %llu, %llu, %llu, %s, %s)", o, x, y, &alpha, co, cx, cy, veda_tensors_get_unary(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_TTS), 0, o, x, y, alpha.U64_U64.x, alpha.U64_U64.y, co, cx, cy, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_unary_ttts(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, VEDAdeviceptr z, const VEDATensors_scalar alpha, const size_t co, const size_t cx, const size_t cy, const size_t cz, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_ADDCMUL:
		case VEDA_TENSORS_UNARY_ADDCDIV:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}
	
	GUARD("unary_ttts(%p, %p, %p, %p, %p, %llu, %llu, %llu, %llu, %s, %s)", o, x, y, z, &alpha, co, cx, cy, cz, veda_tensors_get_unary(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_TTTS), 0, o, x, y, z, alpha.U64_U64.x, alpha.U64_U64.y, co, cx, cy, cz, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_binary(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t oc, const size_t ox, const size_t oy, const VEDATensors_binary_op op, const VEDATensors_dtype dtype) {
	GUARD("binary(%p, %p, %p, %llu, %llu, %llu, %s, %s)", o, x, y, oc, ox, oy, veda_tensors_get_binary(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(BINARY), 0, o, x, y, oc, ox, oy, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_bitwise(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t oc, const size_t ox, const size_t oy, const VEDATensors_bitwise_op op, const VEDATensors_dtype dtype) {
	GUARD("bitwise(%p, %p, %p, %llu, %llu, %llu, %s, %s)", o, x, y, oc, ox, oy, veda_tensors_get_bitwise(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(BITWISE), 0, o, x, y, oc, ox, oy, op, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_count(VEDATensors_chandle handle, VEDAdeviceptr vptr, const size_t elements, const VEDATensors_dtype dtype, size_t* cnt) {
	GUARD("count(%p, %llu, %s)", vptr, elements, veda_tensors_get_dtype(dtype));
	*cnt = 0;
	uint64_t out;
	CVEDA(vedaLaunchKernelEx(KERNEL(COUNT), 0, &out, vptr, elements, veda_tensors_dtype_bytes(dtype)));
	CVEDA(vedaStreamSynchronize(0));
	*cnt = (size_t)out;
	return VEDA_SUCCESS;
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_masked_fill(VEDATensors_chandle handle, VEDAdeviceptr out, const VEDATensors_scalar value, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype) {
	GUARD("masked_fill(%p, %p, %p, %llu, %s)", out, &value, mask, elements, veda_tensors_get_dtype(dtype));
	auto bytes = veda_tensors_dtype_bytes(dtype);
	VEDAdeviceptr ptr;
	CVEDA(vedaMemAllocAsync(&ptr, bytes, 0));
	CVEDA(vedaMemcpyHtoDAsync(ptr, &value, bytes, 0));
	CVEDA(veda_tensors_ll_masked_fill_t(handle, out, ptr, mask, elements, 1, dtype));
	CVEDA(vedaMemFreeAsync(ptr, 0));
	return VEDA_SUCCESS;
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_masked_fill_t(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr value, VEDAdeviceptr mask, const size_t elements, const size_t otherElements, const VEDATensors_dtype dtype) {
	GUARD("masked_fill_t(%p, %p, %p, %llu, %llu, %s)", out, value, mask, elements, otherElements, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(MASKED_FILL), 0, out, value, mask, elements, otherElements, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_masked_scatter(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr in, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype) {
	GUARD("masked_scatter(%p, %p, %p, %llu, %s)", out, in, mask, elements, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(MASKED_SCATTER), 0, out, in, mask, elements, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_masked_select(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr in, VEDAdeviceptr mask, const size_t outElements, const size_t elements, const VEDATensors_dtype dtype) {
	GUARD("masked_select(%p, %p, %p, %llu, %llu, %s)", out, in, mask, outElements, elements, veda_tensors_get_dtype(dtype));
	vedaLaunchKernel(KERNEL(MASKED_SELECT), 0, out, in, mask, outElements, elements, veda_tensors_dtype_bytes(dtype));
	CVEDA(vedaCtxSynchronize());
	return VEDA_SUCCESS;
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_reduce(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr in, const size_t elements, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype) {
	GUARD("reduce(%p, %p, %llu, %s, %s)", out, in, elements, veda_tensors_get_reduce(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(REDUCE), 0, out, in, elements, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_reduce_dim(VEDATensors_chandle handle, VEDAdeviceptr values, VEDAdeviceptr indices, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype) {
	GUARD("reduce_dim(%p, %p, %p, %llu, %llu, %llu, %s, %s)", values, indices, in, left, center, right, veda_tensors_get_reduce(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(REDUCE_DIM), 0, values, indices, in, left, center, right, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_reduce_scalar(VEDATensors_chandle handle, VEDAdeviceptr in, const size_t elements, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype, VEDATensors_scalar* result) {
	GUARD("reduce_scalar(%p, %llu, %s, %s, %p)", in, elements, veda_tensors_get_reduce(op), veda_tensors_get_dtype(dtype), result);
	auto bytes = veda_tensors_dtype_bytes(dtype);
	memset(result, 0, sizeof(VEDATensors_scalar));
	VEDAdeviceptr out;
	CVEDA(vedaMemAllocAsync(&out, bytes, 0));
	CVEDA(veda_tensors_ll_reduce(handle, out, in, elements, op, dtype));
	CVEDA(vedaMemcpyDtoHAsync(result, out, bytes, 0));
	CVEDA(vedaMemFreeAsync(out, 0));
	return vedaStreamSynchronize(0);	
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_convert(VEDATensors_chandle handle, VEDAdeviceptr dst, VEDAdeviceptr src, const size_t elements, const VEDATensors_dtype dstType, const VEDATensors_dtype srcType) {
	GUARD("convert(%p, %p, %llu, %s, %s)", dst, src, elements, veda_tensors_get_dtype(dstType), veda_tensors_get_dtype(srcType));
	return vedaLaunchKernel(KERNEL(CONVERT), 0, dst, src, elements, dstType, srcType);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_transpose(VEDATensors_chandle handle, VEDAdeviceptr dst, VEDAdeviceptr src, const size_t sizeA, const size_t sizeB, const VEDATensors_dtype dtype) {
	GUARD("transpose(%p, %p, %llu, %llu, %s)", dst, src, sizeA, sizeB, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(TRANSPOSE), 0, dst, src, sizeA, sizeB, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_print(VEDATensors_chandle handle, VEDAdeviceptr ptr, const size_t elements, const VEDATensors_dtype dtype) {
	GUARD("print(%p, %llu, %s)", ptr, elements, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(PRINT), 0, ptr, elements, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_ll_cat(VEDATensors_chandle handle, const int inputCnt, const VEDAdeviceptr* inputs, const size_t* dimSizes, const int outputDimCnt, const size_t* outSizes, VEDAdeviceptr output, const int dim, const VEDATensors_dtype dtype) {
	GUARD("cat(%i, %p, %p, %i, %p, %p, %i, %s)", inputCnt, inputs, dimSizes, outputDimCnt, outSizes, output, dim, veda_tensors_get_dtype(dtype));

	std::vector<size_t> offsets(inputCnt);
	std::vector<size_t> center (inputCnt);

	size_t left = 1, right = 1;
	for(int i = 0; i < dim; i++)				left  *= outSizes[i];
	for(int i = dim+1; i < outputDimCnt; i++)	right *= outSizes[i];

	size_t offset = 0;
	for(int i = 0; i < inputCnt; i++) {
		offsets[i]	 = offset;
		center [i]	 = dimSizes[i];
		offset 		+= dimSizes[i];
	}

	auto s_inputs	= sizeof(VEDAdeviceptr) * inputCnt;
	auto s_offsets	= sizeof(size_t) * inputCnt;
	auto s_center	= sizeof(size_t) * inputCnt;

	VEDAdeviceptr d_inputs	= 0;
	VEDAdeviceptr d_offsets	= 0;
	VEDAdeviceptr d_center	= 0;
	CVEDA(vedaMemAllocAsync(&d_inputs,  s_inputs,  0));
	CVEDA(vedaMemAllocAsync(&d_offsets, s_offsets, 0));
	CVEDA(vedaMemAllocAsync(&d_center,  s_center,  0));

	CVEDA(vedaMemcpyHtoDAsync(d_inputs,  inputs,         s_inputs,  0));
	CVEDA(vedaMemcpyHtoDAsync(d_offsets, offsets.data(), s_offsets, 0));
	CVEDA(vedaMemcpyHtoDAsync(d_center,  center .data(), s_center,  0));

	CVEDA(vedaLaunchKernel(KERNEL(CAT), 0, inputCnt, output, d_inputs, d_offsets, d_center, left, right, veda_tensors_dtype_bytes(dtype)));

	CVEDA(vedaMemFreeAsync(d_inputs, 0));
	CVEDA(vedaMemFreeAsync(d_offsets, 0));
	CVEDA(vedaMemFreeAsync(d_center, 0));

	return VEDA_SUCCESS;
}

//------------------------------------------------------------------------------
