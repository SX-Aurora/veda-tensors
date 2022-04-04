// Copyright 2022 NEC Laboratories Europe

#pragma once 

//------------------------------------------------------------------------------
struct VEDATensors_handle_struct {
	VEDAdevice		idx;
	VEDAcontext		ctx;
	VEDAmodule		mod;
	VEDAfunction	kernel[VEDA_TENSORS_KERNEL_CNT];
};
typedef struct VEDATensors_handle_struct* VEDATensors_handle;
typedef const struct VEDATensors_handle_struct* VEDATensors_chandle;

//------------------------------------------------------------------------------
struct VEDATensors_tensor_struct {
	size_t				shape[VEDA_TENSORS_MAX_DIMS];
	size_t				numel;
	int					dims;
	VEDATensors_dtype	dtype;
	VEDAdeviceptr		ptr;

#if __cplusplus
	template<typename T, typename D>
	inline VEDATensors_tensor_struct(const D dims_, const T shape_, const VEDATensors_dtype dtype_, VEDAdeviceptr ptr_) :
		shape	{0},
		numel	(1),
		dims	((int)dims_),
		dtype	(dtype_),
		ptr		(ptr_)
	{
		if(dims) {
			for(int i = 0; i < dims; i++) {
				auto ssize = static_cast<size_t>(shape_[i]);
				shape[i]	 = ssize;
				numel		*= ssize;
			}
		} else { // stupid AI frameworks think that dims == 0 is a scalar...
			dims		= 1;
			shape[0]	= 1;
			numel		= 1;
		}
	}
#endif
};
typedef struct VEDATensors_tensor_struct VEDATensors_tensor;

//------------------------------------------------------------------------------
