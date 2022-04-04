// Copyright 2022 NEC Laboratories Europe

#pragma once

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_binary			(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t oc, const size_t ox, const size_t oy, const VEDATensors_binary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_bitwise			(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t oc, const size_t ox, const size_t oy, const VEDATensors_bitwise_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_cat				(VEDATensors_chandle handle, const int inputCnt, const VEDAdeviceptr* inputs, const size_t* dimSizes, const int outputDimCnt, const size_t* outSizes, VEDAdeviceptr output, const int dim, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_convert			(VEDATensors_chandle handle, VEDAdeviceptr dst, VEDAdeviceptr src, const size_t elements, const VEDATensors_dtype dstType, const VEDATensors_dtype srcType);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_copy			(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t oc, const size_t ox, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_count			(VEDATensors_chandle handle, VEDAdeviceptr vptr, const size_t elements, const VEDATensors_dtype dtype, size_t* cnt);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_fill			(VEDATensors_chandle handle, VEDAdeviceptr ptr, const VEDATensors_scalar value, const size_t cnt, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_masked_select	(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr in, VEDAdeviceptr mask, const size_t outElements, const size_t elements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_masked_scatter	(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr in, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_masked_fill		(VEDATensors_chandle handle, VEDAdeviceptr out, const VEDATensors_scalar value, VEDAdeviceptr mask, const size_t elements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_masked_fill_t	(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr value, VEDAdeviceptr mask, const size_t elements, const size_t otherElements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_prefix_sum		(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr carry, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const VEDATensors_dtype dtype, const int inclusive);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_print			(VEDATensors_chandle handle, VEDAdeviceptr ptr, const size_t elements, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_reduce			(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr in, const size_t elements, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_reduce_dim		(VEDATensors_chandle handle, VEDAdeviceptr values, VEDAdeviceptr indices, VEDAdeviceptr in, const size_t left, const size_t center, const size_t right, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_reduce_scalar	(VEDATensors_chandle handle, VEDAdeviceptr in, const size_t elements, const VEDATensors_reduce_op op, const VEDATensors_dtype dtype, VEDATensors_scalar* result);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_select			(VEDATensors_chandle handle, VEDAdeviceptr out, VEDAdeviceptr in, const size_t ocnt, const size_t ostride, const size_t icnt, const size_t offset, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_transpose		(VEDATensors_chandle handle, VEDAdeviceptr dst, VEDAdeviceptr src, const size_t sizeA, const size_t sizeB, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_unary_b			(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_unary_c			(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_unary_t			(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, const size_t co, const size_t cx, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_unary_tt		(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const size_t co, const size_t cx, const size_t cy, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_unary_tts		(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, const VEDATensors_scalar alpha, const size_t co, const size_t cx, const size_t cy, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);
VEDA_TENSORS_API VEDAresult	veda_tensors_ll_unary_ttts		(VEDATensors_chandle handle, VEDAdeviceptr o, VEDAdeviceptr x, VEDAdeviceptr y, VEDAdeviceptr z, const VEDATensors_scalar alpha, const size_t co, const size_t cx, const size_t cy, const size_t cz, const VEDATensors_unary_op op, const VEDATensors_dtype dtype);

//------------------------------------------------------------------------------