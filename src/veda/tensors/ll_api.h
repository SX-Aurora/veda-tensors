// Copyright 2022 NEC Laboratories Europe

#pragma once

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult	veda_tensors_binary			(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t oc, const size_t ox, const size_t oy, const VEDATensors_binary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_bitwise		(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t oc, const size_t ox, const size_t oy, const VEDATensors_bitwise_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_cat			(VEDATensors_handle handle, const int cnt, const VEDAdeviceptr* inputs, const uint64_t* dimSizes, const uint64_t* outSizes, VEDAdeviceptr output, const int dim, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_convert		(VEDATensors_handle handle, VEDAdeviceptr dst, VEDAdeviceptr src, const size_t elements, const VEDATensors_dtype dstType, const VEDATensors_dtype srcType);
VEDA_TENSORS_API VEDAresult	veda_tensors_copy			(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t oc, const size_t ox, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_count			(VEDATensors_handle handle, VEDAdeviceptr vptr, const size_t elements, const VEDATensors_dtype dtype, size_t* cnt);
VEDA_TENSORS_API VEDAresult	veda_tensors_fill			(VEDATensors_handle handle, VEDAdeviceptr ptr, const void* value, const size_t cnt, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_maskedSelect	(VEDATensors_handle handle, VEDAdeviceptr out, VEDAdeviceptr in, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_masked_Scatter	(VEDATensors_handle handle, VEDAdeviceptr out, VEDAdeviceptr in, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_masked_fill	(VEDATensors_handle handle, VEDAdeviceptr out, const void* value, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_masked_fill_t	(VEDATensors_handle handle, VEDAdeviceptr out, VEDAdeviceptr value, VEDAdeviceptr mask, const size_t elements, const size_t otherElements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_prefixSum		(VEDATensors_handle handle, VEDAdeviceptr out, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const VEDATensors_dtype dtype, const int inclusive);
VEDA_TENSORS_API VEDAresult	veda_tensors_print			(VEDATensors_handle handle, VEDAdeviceptr ptr, const size_t elements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_reduce			(VEDATensors_handle handle, VEDAdeviceptr out, VEDAdeviceptr in, const size_t elements, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_reduce_dim		(VEDATensors_handle handle, VEDAdeviceptr values, VEDAdeviceptr indices, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_reduce_scalar	(VEDATensors_handle handle, VEDAdeviceptr in, const size_t elements, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype, void* result);
VEDA_TENSORS_API VEDAresult	veda_tensors_select			(VEDATensors_handle handle, VEDAdeviceptr out, VEDAdeviceptr in, const size_t ocnt, const size_t ostride, const size_t icnt, const size_t offset, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_transpose		(VEDATensors_handle handle, VEDAdeviceptr dst, VEDAdeviceptr src, const size_t sizeA, const size_t sizeB, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_b		(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_c		(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_t		(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_tt		(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t co, const size_t cx, const size_t cy, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_tts		(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const void* alpha, const size_t co, const size_t cx, const size_t cy, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_ttts		(VEDATensors_handle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, VEDAdeviceptr z, const void* alpha, const size_t co, const size_t cx, const size_t cy, const size_t cz, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);

//------------------------------------------------------------------------------