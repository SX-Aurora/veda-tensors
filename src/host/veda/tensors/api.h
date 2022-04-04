// Copyright 2022 NEC Laboratories Europe

#pragma once

#if __cplusplus 
#define VEDA_TENSORS_API extern "C"
#else
#define VEDA_TENSORS_API
#endif

#include <stdint.h>
#include <stddef.h> // size_t

#include <veda_types.h>

#include <veda/tensors/version.h>
#include <veda/tensors/macros.h>
#include <veda/tensors/enums.h>
#include <veda/tensors/types.h>
#include <veda/tensors/scalar.h>
#include <veda/tensors/init.h>
#include <veda/tensors/get.h>
#include <veda/tensors/low_level.h>
#include <veda/tensors/high_level.h>
