// Copyright 2017-2022 NEC Laboratories Europe

#pragma once

#define FVEDA(...) veda_tensors_check(__VA_ARGS__, __FILE__, __LINE__)
void veda_tensors_check(VEDAresult result, const char* file, const int line);
void veda_tensors_nop(void);
