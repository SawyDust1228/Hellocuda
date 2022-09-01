
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory>

#include "kernel.cuh"
#include "Matrix.h"

#include "torch/torch.h"
#define _DEBUG
using namespace at;

extern "C"
void vector_add(float* v1, float* v2, float* result, int n);

extern "C"
void blur(std::vector<std::vector<float>>& matrix, std::vector<std::vector<float>>& result);

extern "C"
void matrixMultiply(Matrix A, Matrix B, Matrix C);

extern "C"
void viewCudaDeviceInfo();

extern "C"
void conv1d(float* v, float* result, float* m, int n, int k);

extern "C" 
void mergeSort(float* vector, int n);

extern "C"
void vector_sum(const float* vector, int n, float* result);

int main() {
    float result;
    int n = 10000;
    auto a = torch::ones({1, n}, torch::kFloat);
    vector_sum(a.data_ptr<float>(), n, &result);
    // std::cout << a.sum().item<float>() << std::endl;
    std::cout << result << std::endl;
}

// int main() {
//     int n = 20;
//     auto a = torch::randn({1, n});
//     std::cout << a << std::endl;
//     mergeSort(a.data_ptr<float>(), n);
//     std::cout << a << std::endl;
// }

// int main() {
//     int size = 10 * sizeof(float);
//     float* a;
//     float* b;
//     float* result;
//     result = (float*) malloc(size);
//     a = (float*) malloc(size);
//     b = (float*) malloc(size);

//     for(int i = 0; i < 10; i++) {
//         a[i] = 1;
//         b[i] = 2;
//     }
    
//     vector_add(a, b, result, 10);
//     for(int i = 0; i < 10; i++) {
//         printf("[RESULT %d ] : %f\n", i, result[i]);
//     }

//     free(a);free(b);free(result);

//     return 0;
// }


// struct Conv1dNet : torch::nn::Module
// {
//     Conv1dNet(int k) 
//     :conv1(register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(1, 1, k).stride(1).padding((k - 1) / 2).bias(false))))
//     { }
//     torch::Tensor forward(torch::Tensor const& input) {
//         auto x = conv1(input);
//         return x;
//     }
    
//     torch::nn::Conv1d conv1{nullptr};
// };



// int main() {
//     // viewCudaDeviceInfo();
//     int k = 5;
//     auto a = torch::randn({1, 20});

//     auto net = std::make_shared<Conv1dNet>(Conv1dNet(k));
// #ifdef _DEBUG
//     std::cout << *net << std::endl;
// #endif
//     auto result_net = net->forward(a);
//     std::cout << result_net << std::endl;

//     for(auto const& para : net->parameters()) {
//         std::cout << para << std::endl;
//         auto result = torch::zeros_like(a);
//         conv1d(a.data_ptr<float>(), result.data_ptr<float>(), para.data_ptr<float>(), 20, 5);
//         std::cout << result << std::endl;
//     }

//     // vector_add(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), 20);
    
//     return 0;
// }

// void print(std::vector<std::vector<float>> const& v) {
//     std::cout << "[";
//     for(int i = 0; i < v.size(); i++) {
//         std::cout << "[";
//         for(int j = 0 ; j < v[0].size(); j++) {
//             if(j < v[0].size() - 1) {
//                 std::cout << v[i][j] << ", "; 
//             }else{
//                 std::cout << v[i][j]; 
//             }
           
//         }
//         if(i < v.size() - 1) {
//             std::cout << " ]" << std::endl;
//         } else {
//             std::cout << " ]";
//         }
        
//     }
//     std::cout << "]" << std::endl;
// }

// int main() {
//     int h = 10, w = 20;
//     std::vector<std::vector<float>> matrix, result;
//     for(int i = 0 ; i < h; i++) {
//         std::vector<float> v;
//         matrix.push_back(v);
//         result.push_back(v);
//         for(int j = 0 ; j < w; j++) {
//             matrix[i].push_back(1.);
//         }
//     }
//     print(matrix);
//     blur(matrix, result);
//     print(result);
// }


// int main() {
//     Matrix A(10, 20);
//     Matrix B(20, 10);
//     Matrix C(A.m, B.n);

//     A.initialElements();
//     B.initialElements();
//     C.initialElements();

//     matrixMultiply(A, B, C);

//     for(int i = 0; i < C.m; i++) {
//         for(int j = 0; j < C.n; j++) {
//             std::cout << C.elements[i * C.n + j] << " "; 
//         }
//         std::cout << std::endl;
//     }

//     delete[] A.elements;
//     delete[] B.elements;
//     delete[] C.elements;

//     return 0;
// }