// Copyright 2022 NEC Laboratories Europe

#pragma once

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult	veda_tensors_binary			(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, const VEDATensors_binary_op op);
VEDA_TENSORS_API VEDAresult	veda_tensors_bitwise		(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, const VEDATensors_bitwise_op op);
VEDA_TENSORS_API VEDAresult	veda_tensors_cat			(VEDATensors_chandle handle, const int inputCnt, VEDATensors_tensor* inputs, VEDATensors_tensor* output, const int dim);
VEDA_TENSORS_API VEDAresult	veda_tensors_convert		(VEDATensors_chandle handle, VEDATensors_tensor* dst, VEDATensors_tensor* src);
VEDA_TENSORS_API VEDAresult	veda_tensors_copy			(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x);
VEDA_TENSORS_API VEDAresult	veda_tensors_count			(VEDATensors_chandle handle, VEDATensors_tensor* in, size_t* cnt);
VEDA_TENSORS_API VEDAresult	veda_tensors_fill			(VEDATensors_chandle handle, VEDATensors_tensor* ptr, const VEDATensors_scalar scalar);
VEDA_TENSORS_API VEDAresult	veda_tensors_masked_select	(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* in, VEDATensors_tensor* mask);
VEDA_TENSORS_API VEDAresult	veda_tensors_masked_scatter	(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* in, VEDATensors_tensor* mask);
VEDA_TENSORS_API VEDAresult	veda_tensors_masked_fill	(VEDATensors_chandle handle, VEDATensors_tensor* out, const VEDATensors_scalar scalar, VEDATensors_tensor* mask);
VEDA_TENSORS_API VEDAresult	veda_tensors_masked_fill_t	(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* value, VEDATensors_tensor* mask);
VEDA_TENSORS_API VEDAresult	veda_tensors_prefix_sum		(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* carry, VEDATensors_tensor* in, const int axis, const int inclusive);
VEDA_TENSORS_API VEDAresult	veda_tensors_print			(VEDATensors_chandle handle, VEDATensors_tensor* ptr);
VEDA_TENSORS_API VEDAresult	veda_tensors_reduce			(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* in, const VEDATensors_reduce_op op);
VEDA_TENSORS_API VEDAresult	veda_tensors_reduce_dim		(VEDATensors_chandle handle, VEDATensors_tensor* values, VEDATensors_tensor* indices, VEDATensors_tensor* in, const VEDATensors_reduce_op op, const int axis);
VEDA_TENSORS_API VEDAresult	veda_tensors_reduce_scalar	(VEDATensors_chandle handle, VEDATensors_tensor* in, const VEDATensors_reduce_op op, VEDATensors_scalar* result);
VEDA_TENSORS_API VEDAresult	veda_tensors_select			(VEDATensors_chandle handle, VEDATensors_tensor* out, VEDATensors_tensor* in, const size_t ocnt, const size_t ostride, const size_t icnt, const size_t offset);
VEDA_TENSORS_API VEDAresult	veda_tensors_transpose		(VEDATensors_chandle handle, VEDATensors_tensor* dst, VEDATensors_tensor* src);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_b		(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, const VEDATensors_unary_op op);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_c		(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, const VEDATensors_unary_op op);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_t		(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, const VEDATensors_unary_op op);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_tt		(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, const VEDATensors_unary_op op);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_tts		(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, const VEDATensors_scalar alpha, const VEDATensors_unary_op op);
VEDA_TENSORS_API VEDAresult	veda_tensors_unary_ttts		(VEDATensors_chandle handle, VEDATensors_tensor* o, VEDATensors_tensor* x, VEDATensors_tensor* y, VEDATensors_tensor* z, const VEDATensors_scalar alpha, const VEDATensors_unary_op op);
VEDA_TENSORS_API void		veda_tensors_debug			(VEDATensors_tensor* tensor);

//------------------------------------------------------------------------------