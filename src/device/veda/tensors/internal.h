// Copyright 2022 NEC Laboratories Europe

#pragma once 

#include "device.h"

#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <cassert>

#include <veda/omp.h>
#include <tungl/c.h>

#include "check.h"
#include "types.h"
#include "prefixsum.h"
#include "kernel.h"
#include "xyz.h"
