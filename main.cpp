
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

struct node;

extern "C"
void BFS(std::vector<std::vector<int>> const& graph, std::vector<int> const& values, int* result);

extern "C"
void vector_add_new(float* vector, float* result, int n);

extern "C"
void FFT1D(float* vector , float* real, float* image, int n);

extern "C"
void FFTCONV1D(float* vector , float* kernel, float* result, int k , int n);

extern "C"
void FFTCONV2D(float* m1, float* m2 , float* result, int m, int n, int k);

extern "C"
void MatrixElementMult(float* m1, float* m2, float* result, int m, int n);

struct Conv1dNet : torch::nn::Module
{
    Conv1dNet(int k) 
    :conv1(register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(1, 1, k).stride(1).padding((k - 1) / 2).bias(false))))
    {
        for(auto& para : this->parameters()) {
            for(int i = 0; i < k; ++i) {
                auto ptr = para.data_ptr<float>();
                *(ptr + i) = 1.;
            }
        }
    }
    torch::Tensor forward(torch::Tensor const& input) {
        auto x = conv1(input);
        return x;
    }
    
    torch::nn::Conv1d conv1{nullptr};
};

int main() {
    viewCudaDeviceInfo();
    // int n = 20;
    // int k = 5;
    // auto net = std::make_shared<Conv1dNet>(Conv1dNet(k));
    // auto a = torch::ones({1, n});
    // auto kernel = torch::ones({1, k});
    // auto result_net = net->forward(a);

    // std::cout << "Pytorch Result : " <<result_net << std::endl;
    // auto result = torch::zeros_like(a);
    // conv1d(a.data_ptr<float>(), result.data_ptr<float>(), kernel.data_ptr<float>(), n, k);
    // std::cout << "Simple Conv Result" <<result << std::endl;

    // FFTCONV1D(a.data_ptr<float>(), kernel.data_ptr<float>(), result.data_ptr<float>(), k, n);
    // std::cout << "FFT Result" <<result << std::endl;

    auto matrix = torch::ones({10, 10});
    auto matrix2 = 2 * torch::ones({10, 10});
    auto kernel55 = torch::ones({5, 5});
    auto result_matrix = torch::ones_like(matrix);
    // std::cout  << matrix << std::endl;

    MatrixElementMult(matrix.data_ptr<float>(), matrix2.data_ptr<float>(), result_matrix.data_ptr<float>(), 10, 10);
    std::cout << result_matrix << std::endl;

    FFTCONV2D(matrix.data_ptr<float>(), kernel55.data_ptr<float>(), result_matrix.data_ptr<float>(), 10, 10, 5);
    std::cout << result_matrix << std::endl;

    return 0;
}

// int main() {

//     int n = 10;
//     float result;
//     auto a = torch::ones({1, n}, torch::kFloat);
//     vector_add_new(a.data_ptr<float>(), &result, n);
//     return 0;
// }

// int main() {
//     int n = 10;
//     std::vector<int> values;
//     std::vector<std::vector<int>> graph;
//     for(int i = 0; i < n; ++i) {
//         values.push_back(1);
//         graph.push_back(std::vector<int>());
//         for(int j = 0 ; j < n; j++) {
//             if(i != j) {
//                 graph[i].push_back(j);
//             }
//         }
//     }

//     int result = 0;
//     BFS(graph, values, &result);

//     std::cout << result << std::endl;
    
// }

// int main() {
//     float result;
//     int n = 10000;
//     auto a = torch::ones({1, n}, torch::kFloat);
//     vector_sum(a.data_ptr<float>(), n, &result);
//     // std::cout << a.sum().item<float>() << std::endl;
//     std::cout << result << std::endl;
// }

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