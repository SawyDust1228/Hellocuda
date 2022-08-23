
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "kernel.cuh"
#include "Matrix.h"

extern "C"
void vector_add(float* v1, float* v2, float* result, int n);

extern "C"
void blur(std::vector<std::vector<float>>& matrix, std::vector<std::vector<float>>& result);

extern "C"
void matrixMultiply(Matrix A, Matrix B, Matrix C);

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


int main() {
    Matrix A(10, 20);
    Matrix B(20, 10);
    Matrix C(A.m, B.n);

    A.initialElements();
    B.initialElements();
    C.initialElements();

    matrixMultiply(A, B, C);

    for(int i = 0; i < C.m; i++) {
        for(int j = 0; j < C.n; j++) {
            std::cout << C.elements[i * C.n + j] << " "; 
        }
        std::cout << std::endl;
    }

    delete[] A.elements;
    delete[] B.elements;
    delete[] C.elements;

    return 0;
}