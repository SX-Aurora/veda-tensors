// Copyright 2022 NEC Laboratories Europe

#pragma once

#if __cplusplus 
#define VEDA_TENSORS_API extern "C"
#else
#define VEDA_TENSORS_API
#endif

#include <veda_types.h>
#include <veda/tensors/types.h>
#include <veda/tensors/init.h>
#include <veda/tensors/get.h>
#include <veda/tensors/ll_api.h>

