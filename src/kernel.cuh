
#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include <stdio.h>
#include <iostream>
#include <vector>

#include "Matrix.h"

__global__ void vector_add_device(float* v1, float* v2, float* result, int n);

extern "C" 
void vector_add(float* v1, float* v2, float* result, int n);

__global__ void blur_kernel(float* matrix, float* result, int w, int h, int window);

extern "C"
void blur(std::vector<std::vector<float>>& matrix, std::vector<std::vector<float>>& result);

__global__ void matrixMultiply_kernel(Matrix A, Matrix B, Matrix C);

extern "C"
void matrixMultiply(Matrix A, Matrix B, Matrix C);

extern "C"
void viewCudaDeviceInfo();

template<typename T>
__global__ void conv1d_kernel(T* v, T* result, T* m, int n, int k);


template<typename T>
__global__ void conv1d_kernel_constant(T* v, T* result, int n, int k);

extern "C"
void conv1d(float* v, float* result, float* m, int n, int k);

template<typename T>
__global__ void mergeSort_kernel(T* v, T*temp, int n);


extern "C" void mergeSort(float* vector, int n);





#endif