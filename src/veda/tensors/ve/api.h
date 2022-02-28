// Copyright 2022 NEC Laboratories Europe

#pragma once 

#ifndef L_MODULE
#define L_MODULE "veda-tensors"
#endif

#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <cassert>
#include <cstdint>

#include <veda/tensors/types.h>
#include <veda_device.h>
#include <veda_device_omp.h>

#include "check.h"
#include "omp.h"
#include "prefixsum.h"
#include "kernel.h"
#include "xyz.h"
