#include <veda/tensors/api.h>

//------------------------------------------------------------------------------
#define KERNEL(N) handle->kernel[VEDA_TENSORS_KERNEL_##N]
#define GUARD(MSG, ...)\
	if(!handle) return VEDA_ERROR_INVALID_MODULE;\
	L_TRACE("[VE#%i] " MSG, handle->idx, __VA_ARGS__);\
	Guard __guard__(handle)

//------------------------------------------------------------------------------
struct Guard {
	inline Guard(VEDATensors_handle handle) {
		CVEDA(vedaCtxPushCurrent(handle->ctx));
	}

	inline ~Guard(void) {
		VEDAcontext ctx;
		CVEDA(vedaCtxPopCurrent(&ctx));
	}
};

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_copy(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t oc, const size_t ox, const VEDATensors_dtype dtype) {
	GUARD("copy(%p, %p, %llu, %llu, %s)", o, x, oc, ox, veda_tensors_get_dtype(dtype));
	auto bytes = veda_tensors_dtype_bytes(dtype);
	if(bytes == 0)	return VEDA_ERROR_NOT_IMPLEMENTED;
	if(ox == 1)		return vedaLaunchKernel(KERNEL(COPY), 0, o, x, oc, bytes);
	if(oc == ox)	return VEDA_ERROR_INVALID_ARGS;
	return vedaMemcpyDtoD(o, x, oc * bytes);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_fill(VEDAdeviceptr ptr, const void* value, const size_t cnt, const VEDATensors_dtype dtype) {
	GUARD("fill(%p, %p, %llu)", ptr, value, cnt);
	switch(veda_tensors_dtype_bytes(dtype)) {
		case 1:		return vedaMemsetD8Async	(ptr, *(const int8_t*) value, cnt, 0));	break;
		case 2:		return vedaMemsetD16Async	(ptr, *(const int16_t*)value, cnt, 0));	break;
		case 4:		return vedaMemsetD32Async	(ptr, *(const int32_t*)value, cnt, 0));	break;
		case 8:		return vedaMemsetD64Async	(ptr, *(const int64_t*)value, cnt, 0));	break;
		case 16: {
			auto value_ = (const int64_t*)value;
			return vedaMemsetD128Async(ptr, value_[0], value_[1], cnt, 0));	
		} break;
	}
	return VEDA_ERROR_NOT_IMPLEMENTED;
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_prefixSum(VEDAdeviceptr out, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const VEDATensors_dtype dtype, const int inclusive) {
	GUARD("prefix_sum(%p %p, %llu, %llu, %llu, %s, %s)", out, in, left, center, veda_tensor_get_dtype(dtype), inclusive ? "inclusive" : "exclusive");
	return vedaLaunchKernel(KERNEL(PREFIX_SUM), 0, out, in, left, center, right, dtype, inclusive);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_select(VEDAdeviceptr out, VEDAdeviceptr in, const size_t ocnt, const size_t ostride, const size_t icnt, const size_t offset, const VEDATensors_dtype dtype) {
	GUARD("select(%p, %p, %llu, %llu, %llu, %llu, %s)", out, in, ocnt, ostride, icnt, offset, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(SELECT), 0, out, in, ocnt, ostride, icnt, offset, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_b(VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_OP_ISFINITE:
		case VEDA_TENSORS_UNARY_OP_ISINF:
		case VEDA_TENSORS_UNARY_OP_ISNAN:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}
	
	GUARD("unary_b(%p, %p, %llu, %llu, %s, %s)", o, x, co, cx, veda_tensors_get_unary_op(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_B), 0, o, x, co, cx, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_t(VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_OP_ABS:
		case VEDA_TENSORS_UNARY_OP_FLOOR:
		case VEDA_TENSORS_UNARY_OP_CEIL:
		case VEDA_TENSORS_UNARY_OP_RECIPROCAL:
		case VEDA_TENSORS_UNARY_OP_SQRT:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}

	GUARD("unary_t(%p, %p, %llu, %llu, %s, %s)", o, x, co, cx, veda_tensors_get_unary_op(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_T), 0, o, x, co, cx, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_c(VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_OP_ABS:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}

	switch(dtype) {
		case VEDATensors_dtype::F32_F32:
		case VEDATensors_dtype::F64_F64:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}

	GUARD("unary_c(%p, %p, %llu, %llu, %s, %s)", o, x, co, cx, veda_tensors_get_unary_op(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_C), 0, o, x, co, cx, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_tt(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t co, const size_t cx, const size_t cy, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_OP_MIN:
		case VEDA_TENSORS_UNARY_OP_MAX:
		case VEDA_TENSORS_UNARY_OP_ADD:
		case VEDA_TENSORS_UNARY_OP_SUB:
		case VEDA_TENSORS_UNARY_OP_DIV:
		case VEDA_TENSORS_UNARY_OP_MUL:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}
	
	GUARD("unary_tt(%p, %p, %p, %llu, %llu, %llu, %s, %s)", o, x, y, co, cx, cy, veda_tensors_get_unary_op(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_TT), 0, o, x, y, co, cx, cy, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_tts(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const void* alpha, const size_t co, const size_t cx, const size_t cy, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_OP_ADD:
		case VEDA_TENSORS_UNARY_OP_SUB:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}

	GUARD("unary_tts(%p, %p, %p, %p, %llu, %llu, %llu, %s, %s)", o, x, y, alpha, co, cx, cy, veda_tensors_get_unary_op(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_TTS), 0, o, x, y, VEDAstack(alpha, VEDA_ARGS_INTENT_IN, veda_tensor_get_bytes(dtype)), co, cx, cy, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_unary_ttts(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, VEDAdeviceptr z, const void* alpha, const size_t co, const size_t cx, const size_t cy, const size_t cz, const VEDATensors_unary_op op, const VEDATensors_dtype dtype) {
	switch(op) {
		case VEDA_TENSORS_UNARY_OP_ADDCMUL:
		case VEDA_TENSORS_UNARY_OP_ADDCDIV:	break;
		default:	return VEDA_ERROR_NOT_IMPLEMENTED;
	}
	
	GUARD("unary_ttts(%p, %p, %p, %p, %p, %llu, %llu, %llu, %llu, %s, %s)", o, x, y, z, alpha, co, cx, cy, cz, veda_tensors_get_unary_op(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(UNARY_TTTS), 0, o, x, y, z, VEDAstack(alpha, VEDA_ARGS_INTENT_IN, veda_tensor_get_bytes(dtype)), co, cx, cy, cz, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_binary(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t oc, const size_t ox, const size_t oy, const VEDATensors_binary_op op, const VEDATensors_dtype dtype) {
	GUARD("binary(%p, %p, %p, %llu, %llu, %llu, %s, %s)", o, x, y, oc, ox, oy, veda_tensors_get_bitwise(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(BINARY), 0, o, x, y, oc, ox, oy, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_bitwise(VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t oc, const size_t ox, const size_t oy, const VEDATensors_bitwise_op op, const VEDATensors_dtype dtype) {
	GUARD("bitwise(%p, %p, %p, %llu, %llu, %llu, %s, %s)", o, x, y, oc, ox, oy, veda_tensors_get_bitwise(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(BITWISE), 0, o, x, y, oc, ox, oy, op, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_count(VEDAdeviceptr vptr, const size_t elements, const VEDATensors_dtype dtype, size_t* cnt) {
	GUARD("count(%p, %llu, %s)", vptr, elements, veda_tensors_get_dtype(dtype));
	*cnt = 0;
	uint64_t out;
	CVEDA(vedaLaunchKernelEx(KERNEL(COUNT), 0, &out, vptr, elements, veda_tensors_dtype_bytes(dtype));
	CVEDA(vedaStreamSynchronize(0));
	*cnt = (size_t)out;
	return VEDA_SUCCESS;
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_masked_fill(VEDAdeviceptr out, const void* value, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype) {
	GUARD("masked_fill(%p, %p, %p, %llu, %s)", out, value, mask, elements, veda_tensors_get_dtype(dtype));
	auto bytes = veda_tensors_dtype_bytes(dtype);
	VEDAdeviceptr ptr;
	CVEDA(vedaMemAllocAsync(&ptr, bytes, 0));
	CVEDA(vedaMemcpyHtoDAsync(ptr, value, bytes, 0));
	CVEDA(veda_tensors_masked_fill_t(out, ptr, mas, elements, 1, dtype));
	CVEDA(vedaMemFreeAsync(ptr, 0));
	return VEDA_SUCCESS;
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_masked_fill_t(VEDAdeviceptr out, VEDAdeviceptr value, VEDAdeviceptr mask, const size_t elements, const size_t otherElements, const VEDATensors_dtype dtype) {
	GUARD("masked_fill_t(%p, %p, %p, %llu, %llu, %s)", out, value, mask, elements, otherElements, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(MASKED_FILL_TENSOR), 0, out, value, mask, elements, otherElements, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_masked_Scatter(VEDAdeviceptr out, VEDAdeviceptr in, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype) {
	GUARD("masked_scatter(%p, %p, %p, %llu, %s)", out, in, mask, elements, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(MASKED_SCATTER), 0, out, in, mask, elements, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_maskedSelect(VEDAdeviceptr out, VEDAdeviceptr in, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype) {
	GUARD("masked_select(%p, %p, %p, %llu, %s)", out, in, mask, elements, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(MASKED_SELECT), 0, out, in, mask, elements, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_reduce(VEDAdeviceptr out, VEDAdeviceptr in, const size_t elements, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype) {
	GUARD("reduce(%p, %p, %llu, %s, %s)", out, in, elements, veda_tensors_get_reduce(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(REDUCE), 0, out, in, elements, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_reduce_dim(VEDAdeviceptr values, VEDAdeviceptr indices, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype) {
	GUARD("reduce_dim(%p, %p, %p, %llu, %llu, %llu, %s, %s)", values, indices, in, left, center, right, veda_tensors_get_reduce(op), veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(REDUCE_DIM), 0, values, indices, in, left, center, right, op, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_reduce_scalar(VEDAdeviceptr in, const size_t elements, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype, void* result) {
	GUARD("reduce_scalar(%p, %llu, %s, %s, %p)", in, elements, veda_tensors_get_reduce(op), veda_tensors_get_dtype(dtype), result);
	auto bytes = veda_tensors_dtype_bytes(dtype);
	VEDAdeviceptr out;
	CVEDA(vedaMemAllocAsync(&out, bytes, 0));
	CVEDA(vedaLaunchKernel(KERNEL(REDUCE_SCALAR), 0, out, in, elements, op, dtype));
	CVEDA(vedaMemCpyDtoHAsync(result, out, bytes, 0));
	CVEDA(vedaMemFreeAsync(out, 0));
	return vedaStreamSynchronize(0);	
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_convert(VEDAdeviceptr dst, VEDAdeviceptr src, const size_t elements, const VEDATensors_dtype dstType, const VEDATensors_dtype srcType) {
	GUARD("convert(%p, %p, %llu, %s, %s)", dst, src, elements, veda_tensors_get_dtype(dstType), veda_tensors_get_dtype(srcType));
	return vedaLaunchKernel(KERNEL(CONVERT), 0, dst, src, elements, dstType, srcType);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_transpose(VEDAdeviceptr dst, VEDAdeviceptr src, const size_t sizeA, const size_t sizeB, const VEDATensors_dtype dtype) {
	GUARD("transpose(%p, %p, %llu, %llu, %s)", dst, src, sizeA, sizeB, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(TRANSPOSE), 0, dst, src, sizeA, sizeB, veda_tensors_dtype_bytes(dtype));
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_print(VEDAdeviceptr ptr, const size_t elements, const VEDATensors_dtype dtype) {
	GUARD("print(%p, %llu, %s)", ptr, elements, veda_tensors_get_dtype(dtype));
	return vedaLaunchKernel(KERNEL(PRINT), 0, ptr, elements, dtype);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_cat(const int cnt, const VEDAdeviceptr* inputs, const uint64_t* dimSizes, const uint64_t* outSizes, VEDAdeviceptr output, const int dim, const VEDATensors_dtype dtype) {
	L_TRACE(renderDevice(this) << "cat(["
		<< ffor(auto p : inputs,   g << (void*)p;	) << "], ["
		<< ffor(auto i : dimSizes, g << i; 			) << "], ["
		<< ffor(auto i : outSizes, g << i; 			) << "], "
		<< (void*)output << ", " << dim << ", " << dtype << ")");
	Guard __guard__(this);

	auto inputCnt	= (int)inputs.size();
	auto dims		= (int)outSizes.size();

	std::vector<int> offsets(inputs.size());
	std::vector<int> center (inputs.size());

	int left = 1, right = 1;
	for(int i = 0; i < dim; i++)
		left *= outSizes[i];

	for(int i = dim+1; i < dims; i++)
		right *= outSizes[i];

	int offset = 0;
	for(int i = 0; i < inputCnt; i++) {
		offsets[i]	= offset;
		center[i]	= (int)dimSizes[i];
		offset 		+= (int)dimSizes[i];
	}

	L_TRACE(renderDevice(this) << "sol_ve_cat(" << (void*)output << ", " << inputCnt << ", ["
		<< ffor(auto p : inputs,	g << (void*)p;	) << "], ["
		<< ffor(auto i : offsets,	g << i;			) << "], ["
		<< ffor(auto i : center,	g << i;			) << "], "
		<< left << ", " << right << ", " << dtype.bits() << ")");

	auto s_inputs	= sizeof(VEDAdeviceptr) * inputs.size();
	auto s_offsets	= sizeof(int) * offsets.size();
	auto s_center	= sizeof(int) * center.size();

	VEDAdeviceptr d_inputs	= 0;
	VEDAdeviceptr d_offsets	= 0;
	VEDAdeviceptr d_center	= 0;
	CVEDA(vedaMemAllocAsync(&d_inputs,  s_inputs,  0));
	CVEDA(vedaMemAllocAsync(&d_offsets, s_offsets, 0));
	CVEDA(vedaMemAllocAsync(&d_center,  s_center,  0));

	CVEDA(vedaMemcpyHtoDAsync(d_inputs,  inputs .data(), s_inputs, 0));
	CVEDA(vedaMemcpyHtoDAsync(d_offsets, offsets.data(), s_offsets, 0));
	CVEDA(vedaMemcpyHtoDAsync(d_center,  center .data(), s_center, 0));

	CVEDA(vedaLaunchKernel(func(Kernel::Cat), 0, inputCnt, output, d_inputs, d_offsets, d_center, left, right, dtype.bits()));

	CVEDA(vedaMemFreeAsync(d_inputs, 0));
	CVEDA(vedaMemFreeAsync(d_offsets, 0));
	CVEDA(vedaMemFreeAsync(d_center, 0));
}

//------------------------------------------------------------------------------
#include "__ns.h"
