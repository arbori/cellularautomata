#pragma once

#include "cuda_runtime.h"

cudaError_t BinarySiteEntropy(int* ca, int X, int Y, float* entropyCA);
