// Copyright 2022 NEC Laboratories Europe

#pragma once

VEDA_TENSORS_API	const char*	veda_tensors_get_unary		(const VEDATensors_unary_op op);
VEDA_TENSORS_API	const char*	veda_tensors_get_reduce		(const VEDATensors_reduce_op op);
VEDA_TENSORS_API	const char*	veda_tensors_get_dtype		(const VEDATensors_dtype dtype);
VEDA_TENSORS_API	const char*	veda_tensors_get_binary		(const VEDATensors_binary_op op);
VEDA_TENSORS_API	const char*	veda_tensors_get_bitwise	(const VEDATensors_bitwise_op op);
VEDA_TENSORS_API	const char*	veda_tensors_get_kernel		(const VEDATensors_kernel kernel);
VEDA_TENSORS_API	const char*	veda_tensors_get_kernel_func(const VEDATensors_kernel kernel);
VEDA_TENSORS_API	int			veda_tensors_dtype_bytes	(const VEDATensors_dtype dtype);
