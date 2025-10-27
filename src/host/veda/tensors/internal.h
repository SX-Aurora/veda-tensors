// Copyright 2022-2025 NEC Laboratories Europe

#pragma once

#include "api.h"

#include <string>		// std::string
#include <string_view>	// std::string_view
#include <vector>		// std::vector
#include <cstdlib>		// setenv
#include <cstring>		// memset
#include <dlfcn.h>		// dladdr
#include <tuple>		// std::tuple
#include <map>			// std::map
#include <mutex>		// std::mutex, std::lock_guard

#include <veda.h>

#define L_MODULE "VEDA-TENSORS"
#include <tungl/c.h>