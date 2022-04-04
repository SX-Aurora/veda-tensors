#include "internal.h"
//------------------------------------------------------------------------------
typedef std::map<int, VEDATensors_handle_struct> Map;
static Map s_map;
static std::mutex s_mutex;

//------------------------------------------------------------------------------
inline const char* veda_tensors_lib_path(void) {
	static std::string s_home;
	if(VEDA_TENSORS_UNLIKELY(s_home.empty())) {
		Dl_info dl_info;
		dladdr((void*)&veda_tensors_get_handle_by_ctx, &dl_info);
		s_home = dl_info.dli_fname;
		auto pos = s_home.find_last_of('/'); assert(pos != std::string::npos);
		pos = s_home.find_last_of('/', pos-1); assert(pos != std::string::npos);
		s_home.replace(pos, std::string::npos, "");
		s_home.append("/libve/libveda-tensors.vso");
		L_TRACE("determined lib-path: %s", s_home.c_str());
	}
	return s_home.c_str();
}

//------------------------------------------------------------------------------
static VEDAresult veda_tensors_get_handle_helper(VEDATensors_handle* handle, VEDAcontext ctx, VEDAdevice device) {
	if(handle == 0)	return VEDA_ERROR_INVALID_HANDLE;
	*handle = 0;

	if(device < 0) {
		if(ctx == 0) {
			CVEDA(vedaCtxGetDevice(&device));
		} else {
			CVEDA(vedaCtxPushCurrent(ctx));
			CVEDA(vedaCtxGetDevice(&device));
			CVEDA(vedaCtxPopCurrent(&ctx));
		}
	}

	std::lock_guard<std::mutex> __lock__(s_mutex);
	auto it = s_map.find(device);

	if(VEDA_TENSORS_LIKELY(it != s_map.end())) {
		*handle = &it->second;
	} else {
		if(ctx == 0)
			CVEDA(vedaDevicePrimaryCtxRetain(&ctx, device));
			
		CVEDA(vedaCtxPushCurrent(ctx));
		auto& hnd = s_map[device];
		hnd.ctx = ctx;
		hnd.idx = device;
		
		// Load Module + Kernels into this Handle ------------------------------
		CVEDA(vedaModuleLoad(&hnd.mod, veda_tensors_lib_path()));

		for(int k = 0; k < VEDA_TENSORS_KERNEL_CNT; k++)
			CVEDA(vedaModuleGetFunction(&hnd.kernel[k], hnd.mod, veda_tensors_get_kernel_func((VEDATensors_kernel)k)));

		// Return Handle -------------------------------------------------------
		*handle = &hnd;

		CVEDA(vedaCtxPopCurrent(&ctx));
	}

	return VEDA_SUCCESS;
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_get_handle_by_ctx(VEDATensors_handle* handle, VEDAcontext ctx) {
	return veda_tensors_get_handle_helper(handle, ctx, -1);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_get_handle_by_id(VEDATensors_handle* handle, VEDAdevice device) {
	return veda_tensors_get_handle_helper(handle, 0, device);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_get_handle(VEDATensors_handle* handle) {
	return veda_tensors_get_handle_helper(handle, 0, -1);
}

//------------------------------------------------------------------------------
VEDA_TENSORS_API VEDAresult veda_tensors_destroy_handle(VEDATensors_chandle handle) {
	if(!handle)
		return VEDA_ERROR_INVALID_HANDLE;

	CVEDA(vedaModuleUnload(handle->mod));
	std::lock_guard<std::mutex> lock(s_mutex);
	if(s_map.erase(handle->idx) == 1)
		return VEDA_SUCCESS;
	return VEDA_ERROR_INVALID_HANDLE;
}

//------------------------------------------------------------------------------
