// Copyright 2022 NEC Laboratories Europe

#pragma once

VEDA_TENSORS_API VEDAresult veda_tensors_get_handle			(VEDATensors_handle* handle);
VEDA_TENSORS_API VEDAresult veda_tensors_get_handle_by_ctx	(VEDATensors_handle* handle, VEDAcontext ctx);
VEDA_TENSORS_API VEDAresult veda_tensors_get_handle_by_id	(VEDATensors_handle* handle, VEDAdevice device);
VEDA_TENSORS_API VEDAresult veda_tensors_destroy_handle		(VEDATensors_chandle handle);
