// Copyright 2022 NEC Laboratories Europe

#pragma once

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_create_handle(VEDATensors_handle* handle) {
	*handle = 0;

	VEDAcontext ctx;
	CVEDA(vedaCtxGetCurrent(&ctx));
	
	*handle = new VEDATensors_handle_struct;
	CVEDA(vedaCtxGetDevice(&handle->idx));
	CVEDA(vedaModuleLoad(&handle->mod, "libveda-tensors.vso"));

	for(VEDATensors_kernel k = (VEDATensors_kernel)0; i < VEDA_TENSORS_KERNEL_CNT; k++) {
		CVEDA(vedaModuleGetFunction(&handle->func[k], handle->mod, veda_tensors_get_kernel_func(i)));
	}

	return VEDA_SUCCESS;
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_destroy_handle(VEDATensors_handle handle) {
	if(!handle)
		return VEDA_ERROR_INVALID_ARGS;

	CVEDA(vedaModuleUnload(handle->mod));
	delete handle;

	return VEDA_SUCCESS;
}

//------------------------------------------------------------------------------
