// Copyright 2022-2025 NEC Laboratories Europe

#pragma once 

#include "device.h"

#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <cassert>
#include <chrono>

#include <veda/omp.h>
#include <tungl/c.h>

#define VEDA_TENSORS_NO_VECTOR 		"_NEC novector"
#define VEDA_TENSORS_PACKED_VECTOR	"_NEC packed_vector"
#define VEDA_TENSORS_PACKED_REG		"_NEC pvreg(vreg)"
#define VEDA_TENSORS_PACKED_LEN		512
#define VEDA_TENSORS_VECTOR			"_NEC vector"
#define VEDA_TENSORS_VECTOR_REG		"_NEC vreg(vreg)"
#define VEDA_TENSORS_VECTOR_LEN		256
#define VEDA_TENSORS_FORCE_INLINE	"_NEC always_inline"
#define VEDA_TENSORS_NO_INLINE		"_NEC noinline"
#define VEDA_TENSORS_SELECT_VECTOR	"_NEC select_vector"

#include "time.h"
#include "check.h"
#include "types.h"
#include "prefixsum.h"
#include "kernel.h"
#include "xyz.h"
