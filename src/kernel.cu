
#include "kernel.cuh"
#include <cufft.h>

// #define DEBUG

#define CONST_MEMORY 25
#define MAX_NODES 1000
__constant__ float weight[CONST_MEMORY];
__constant__ node nodes[MAX_NODES];

__global__ void vector_add_device(float* v1, float* v2, float* result, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n) {
        result[id] = v1[id] + v2[id];
        // printf("%f", result[id]);
    }
}

extern "C"
void vector_add(float* v1, float* v2, float* result, int n) {
    int size = sizeof(float) * n;
    float* v1_gpu; float* v2_gpu; float* result_gpu;
    cudaMalloc((void**) &v1_gpu, size);
    cudaMalloc((void**) &v2_gpu, size);
    cudaMalloc((void**) &result_gpu, size);
    cudaMemcpy(v1_gpu, v1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(v2_gpu, v2, size, cudaMemcpyHostToDevice);

    vector_add_device<<<ceil(n / 256.0), 256>>>(v1_gpu, v2_gpu, result_gpu, n);
    cudaMemcpy(result, result_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree(v1_gpu);cudaFree(v2_gpu);cudaFree(result_gpu);
}

__device__ bool is_valid(int id, int w, int h) {
    return (id > -1 && id < w * h) ? true : false;
}

__global__ void blur_kernel(float* matrix, float* result, int w, int h, int window) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int step = (window - 1) / 2;
    int id = idx * w + idy;
    if(!is_valid(id, w, h)) {
        return;
    }
    float value = 0;
    for(int i = -step; i <= step; i++) {
        for(int j = -step; j <= step; j++) {
            int id_window = (idx + i) * w + (idy + j);
            if((idx + i) > -1 && (idx + i) < h && (idy + j) > -1 && (idy + j) < w) {
                value += matrix[id_window];
            }
        }
    }
    result[id] = value;
    // printf("Idx : %d, Idy : %d, value : %.2f\n", idx, idy, value);
}



extern "C"
void blur(std::vector<std::vector<float>>& matrix, std::vector<std::vector<float>>& result) {
    int h = matrix.size();
    int w = matrix[0].size();
    float* matrix_ptr = new float[w * h];
    float* result_ptr = new float[w * h];

    for (size_t i = 0; i < h; i++) {
        for (size_t j = 0; j < w; j++) {
            matrix_ptr[i * w + j] = matrix[i][j];
        }
    }

    float* matrix_gpu; float* result_gpu;
    int size = sizeof(float) * w * h;
    cudaMalloc((void**) &matrix_gpu, size);
    cudaMalloc((void**) &result_gpu, size);

    cudaMemcpy(matrix_gpu, matrix_ptr, size, cudaMemcpyHostToDevice);

    dim3 grid(ceil(h / 16.), ceil(w / 16.), 1);
    dim3 block(16, 16, 1);

    printf("h : %d, w : %d\n", h, w);
    blur_kernel<<<grid, block>>>(matrix_gpu, result_gpu, w, h, 3);
    cudaDeviceSynchronize();

    cudaMemcpy(result_ptr, result_gpu, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            result[i].push_back(result_ptr[i * w + j]);
        } 
    }

    cudaFree(matrix_gpu);
    cudaFree(result_gpu);
    
    delete[] matrix_ptr;
    delete[] result_ptr;

}

__device__ int getIndex(Matrix M, int x, int y) {
    return M.n * x + y;
}

__global__ void matrixMultiply_kernel(Matrix A, Matrix B, Matrix C) {
    assert(A.n == B.m);
    int k = A.n;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx > -1 && idx < C.m && idy > -1 && idy < C.n) {
        float value = 0.;
        for(int i = 0; i < k; i++) {
            value += (A.elements[getIndex(A, idx, i)] * B.elements[getIndex(B, i, idy)]);
        }

        C.elements[getIndex(C, idx, idy)] = value;
        // printf("[IDX] : %d, [IDY] : %d, [VALUE] : %.2f\n", idx, idy, value);
    }


}

extern "C"
void matrixMultiply(Matrix A, Matrix B, Matrix C) {
    int BLOCK_WIDTH = 4;
    Matrix A_gpu(A.m, A.n);
    Matrix B_gpu(B.m, B.n);
    Matrix C_gpu(C.m, C.n);

    size_t size_A = sizeof(float) * A.m * A.n;
    size_t size_B = sizeof(float) * B.m * B.n;
    size_t size_C = sizeof(float) * C.m * C.n;

    cudaMalloc((void **) &(A_gpu.elements), size_A);
    cudaMalloc((void **) &(B_gpu.elements), size_B);
    cudaMalloc((void **) &(C_gpu.elements), size_C);

    cudaMemcpy(A_gpu.elements, A.elements, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu.elements, B.elements, size_B, cudaMemcpyHostToDevice);
    // cudaMemcpy(C_gpu.elements, C.elements, size_C, cudaMemcpyHostToDevice);

    dim3 grid(ceil((C.m + 0.) / BLOCK_WIDTH), ceil((C.n + 0.) / BLOCK_WIDTH));
    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);

    // printf("%d %d %d\n", grid.x, grid.y, grid.z);
    // printf("%d %d %d\n", block.x, block.y, block.z);

    matrixMultiply_kernel<<<grid, block>>>(A_gpu, B_gpu, C_gpu);

    cudaMemcpy(C.elements, C_gpu.elements, size_C, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu.elements);
    cudaFree(B_gpu.elements);
    cudaFree(C_gpu.elements);

}

extern "C"
void viewCudaDeviceInfo() {
    int num_device;
    cudaGetDeviceCount(&num_device);

    cudaDeviceProp prop;
    for(int i = 0; i < num_device; ++i) {
        cudaGetDeviceProperties(&prop, i);
    }

    printf("[NUM_DEVICES] : %d, [MAX_THREAD_PER_BLOCK] : %d, [SHARE_MEMORY_SIZE] : %d\n", num_device, prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    printf("[MAX_BLOCK_X_SIZE] : %d, [MAX_BLOCK_Y_SIZE] : %d, [MAX_BLCOK_Z_SIZE] : %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
}

__device__ bool is_valid_conv1d(int index, int n) {
    if(index > -1 && index < n) {
        return true;
    }
    return false;
}

template<typename T>
__global__ void conv1d_kernel(T* v, T* result, T* m, int n, int k) {
    int step  = (k - 1) / 2;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n) {
        for(int i = -step; i <= step; ++i) {
            if(is_valid_conv1d(i + idx, n)) {
                result[idx] += v[i + idx] * m[step + i];
            }
        }
    }
}


template<typename T>
__global__ void conv1d_kernel_constant(T* v, T* result, int n, int k) {
    int step  = (k - 1) / 2;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n) {
        for(int i = -step; i <= step; ++i) {
            if(is_valid_conv1d(i + idx, n)) {
                result[idx] += v[i + idx] * weight[step + i];
            }
        }
    }
}

extern "C"
void conv1d(float* v, float* result, float* m, int n, int k) {
    assert(k % 2 == 1);
    float* v_gpu;
    float* result_gpu;
    float* m_gpu;
    int size = sizeof(float) * n;
    int size_mask = sizeof(float) * k;

    
    cudaMalloc((void**) &v_gpu, size);
    cudaMalloc((void**) &result_gpu, size);
    cudaMalloc((void**) &m_gpu, size_mask);

    cudaMemcpy(v_gpu, v, size, cudaMemcpyHostToDevice);
    cudaMemset(result_gpu, 0., size);
    cudaMemcpyToSymbol(weight, m, size_mask);
    cudaMemcpy(m_gpu, m, size_mask, cudaMemcpyHostToDevice);

    dim3 grid(ceil(n / 256.));
    dim3 block(256);
#ifdef DEBUG
    printf("[GRID X] : %d, [BLOCK X] : %d\n", grid.x, block.x);
#endif
    // conv1d_kernel<<<grid, block>>>(v_gpu, result_gpu, m_gpu, n, k);

    conv1d_kernel_constant<<<grid, block>>>(v_gpu, result_gpu, n, k);
    cudaDeviceSynchronize();
    cudaMemcpy(result, result_gpu, size, cudaMemcpyDeviceToHost);
    
    cudaFree(v_gpu);
    cudaFree(result_gpu);
    cudaFree(m_gpu);
}

template<typename T>
__global__ void mergeSort_kernel(T* v, T* temp, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = 2; i < 2 * n; i *= 2) {
        int len = i;
        if(n - idx < i) {
            len = n - idx;
        }


        if(idx % i == 0) {
            T* subA = &v[idx];
            int lenA = i / 2, k = 0;

            T* subB = &v[idx + lenA];
            int lenB = len - lenA, j = 0;

            int p = idx;
            while (/* condition */ k < lenA && j < lenB)
            {
                /* code */
                if(subA[k] < subB[j]) {
                    temp[p++] = subA[k];
                    k++;
                } else {
                    temp[p++] = subB[j];
                    j++;
                }
            }

            while(k < lenA) {
                temp[p++] = subA[k];
                k++;
            }
            
            while(j < lenB) {
                temp[p++] = subB[j];
                j++;
            }

            for(int m = idx; m < idx + len; m++) {
                v[m] = temp[m];
            }
        }
        __syncthreads();

    }
}

extern "C" 
void mergeSort(float* vector, int n) {
    float* vector_gpu;
    float* temp_gpu;
    int size = sizeof(float) * n;
    cudaMalloc((void**) &vector_gpu, size);
    cudaMalloc((void**) &temp_gpu, size);

    cudaMemcpy(vector_gpu, vector, size, cudaMemcpyHostToDevice);
    cudaMemset(temp_gpu, 0, size);

    dim3 grid(ceil(n / 256.));
    dim3 block(ceil(256));

    mergeSort_kernel<<<grid, block>>>(vector_gpu, temp_gpu, n);
    cudaDeviceSynchronize();

    cudaMemcpy(vector, vector_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree(vector_gpu);
    cudaFree(temp_gpu);
}


template<typename T>
__global__ void vector_sum_kernel(T* vector, int n, T* result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n) {
        for(int i = 2 ; i < 2 * n; i *= 2) {
            if(idx % i == 0) {
                int index = idx + i / 2;
                if(index < n) {
                    vector[idx] += vector[index];
                }
            }
            __syncthreads();
        }
    }
}

extern "C"
void vector_sum(const float* vector, int n, float* result) {
    float* vector_gpu;
    float* v = new float[n];
    int size = sizeof(float) * n;
    cudaMalloc((void **) &vector_gpu, size);
    cudaMemcpy(vector_gpu, vector, size, cudaMemcpyHostToDevice);

    dim3 grid(ceil( n / 256.));
    dim3 block(256);

#ifdef DEBUG
    printf("[GRID X] : %d, [BLOCK X] : %d\n", grid.x, block.x);
#endif

    vector_sum_kernel<<<grid, block>>>(vector_gpu, n, result);
    cudaMemcpy(v, vector_gpu, size, cudaMemcpyDeviceToHost);
#ifdef DEBUG
    printf("[");
    for(int i = 0; i < n; ++i) {
        printf("%.2f, ", v[i]);
    }
    printf("]\n");
#endif
    *result = v[0];
    cudaFree(vector_gpu);
    delete[] v;
}



__global__ void BFS_kernel(int* V, int* E, int* F, int* visited, int num_v, int* result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

#ifdef DEBUG
    // printf("[THREAD_ID] : %d\n", idx);
#endif

    if(idx < num_v) {
        if(F[idx]) {
            F[idx] = 0;
            visited[idx] = 1;
            // *result += nodes[idx].value;

            atomicAdd(result, nodes[idx].value);
            for(int i = V[idx]; i < V[idx + 1]; ++i) {
                if(!visited[E[i]]) {
                    atomicAdd(&F[E[i]], 1) ;
                }
            } 
        }
    }
}

__global__ void is_all_zero(int* vector, int n, int* result) {
    int idx =  blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n) {
        atomicAdd(result, vector[idx]);
    }
}

extern "C"
void BFS(std::vector<std::vector<int>> const& graph, std::vector<int> const& values, int* result) {
    int n = values.size();
    node* ns = new node[values.size()];
    std::vector<int> V, E; V.push_back(0);
    for(int i = 0 ; i < n; ++i) {
        ns[i].id = i;
        ns[i].value = values[i];
        for(auto const& id : graph[i]) {
            E.push_back(id);
        }
        V.push_back(E.size());
    }

    int* V_gpu;
    int* E_gpu;
    int* F_gpu;
    int* visited_gpu;
    int* flag;
    int* result_gpu;
    

    cudaMalloc((void**) &V_gpu, sizeof(int) * V.size());
    cudaMalloc((void**) &E_gpu, sizeof(int) * E.size());
    cudaMallocManaged((void**) &F_gpu, sizeof(int) * n);
    cudaMallocManaged((void**) &visited_gpu, sizeof(int) * n);
    cudaMallocManaged((void**) &flag, sizeof(int));
    cudaMallocManaged((void**) &result_gpu, sizeof(int));

    cudaMemset(F_gpu, 0, sizeof(int) * n); F_gpu[0] = 1;
// #ifdef DEBUG
//     printf("%d\n", F_gpu[0]);
//     for(int i = 0; i < V.size(); i++) {
//         std::cout << V[i] << " ";
//     }
//     std::cout << std::endl;
//     for(int i = 0; i < E.size(); i++) {
//         std::cout << E[i] << " ";
//     }
// #endif
    cudaMemset(visited_gpu, 0, sizeof(int) * n);
    cudaMemcpy(V_gpu, V.data(), sizeof(int) * V.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(E_gpu, E.data(), sizeof(int) * E.size(), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nodes, ns, sizeof(node) * n);

    dim3 grid(ceil(n / 256.));
    dim3 block(256);

#ifdef DEBUG
    printf("[GRID X] : %d, [BLOCK X] : %d\n", grid.x, block.x);
#endif
    *flag = 0;
    *result_gpu = 0;
    is_all_zero<<<grid, block>>>(F_gpu, n, flag);
#ifdef DEBUG
        printf("[FGPU] : %d\n", F_gpu[0]);
        printf("[F_TEMP] : %d\n", *flag);
#endif
    while(*flag != 0) {
#ifdef DEBUG
        printf("[F_TEMP] : %d\n", F_gpu[0]);
#endif
        BFS_kernel<<<grid, block>>>(V_gpu, E_gpu, F_gpu, visited_gpu, n, result_gpu);
        cudaDeviceSynchronize();
        *flag = 0;
        is_all_zero<<<grid, block>>>(F_gpu, n, flag);
    }

    cudaMemcpy(result, result_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    

    cudaFree(V_gpu);
    cudaFree(E_gpu);
    cudaFree(F_gpu);
    cudaFree(visited_gpu);
    cudaFree(flag);
    


    delete[] ns; 
}

__global__ void vector_add_new_kernel(float* vector, float* result, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
// #ifdef DEBUG
//     printf("[ID] : %d\n", idx);
// #endif
    if(idx < n) {
        // *result += vector[idx];
        atomicAdd(result, vector[idx]);
    }
}

extern "C"
void vector_add_new(float* vector, float* result, int n) {
    *result = 0;
    float* vector_gpu;
    float* result_gpu;

    cudaMallocManaged((void**) &result_gpu, sizeof(float));
    *result_gpu = 0.;
    cudaMalloc((void**) &vector_gpu, sizeof(float) * n);
    cudaMemcpy(vector_gpu, vector, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(ceil(n / 256.)), block(256);
#ifdef DEBUG
    printf("[GRID X] : %d, [BLOCK X] : %d\n", grid.x, block.x);
#endif
    vector_add_new_kernel<<<grid, block>>>(vector_gpu, result_gpu, n);
    cudaDeviceSynchronize();
    cudaMemcpy(result, result_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    printf("[RESULT] : %.2f", *result);

    cudaFree(vector_gpu);
    cudaFree(result_gpu);
}


__global__ void realToComplex(float* vector, cufftComplex* vector_complex, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n) {
        vector_complex[idx].x = vector[idx];
        vector_complex[idx].y = 0.0;
    }
}

__global__ void complexToCPU(cufftComplex* vector_complex, float* real, float* image, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n) {
        real[idx] = vector_complex[idx].x;
        image[idx] = vector_complex[idx].y;
    }
}

extern "C"
void FFT1D(float* vector , float* real, float* image, int n) {
    float* vector_gpu, *real_gpu, *image_gpu;
    cudaMalloc((void**) &vector_gpu, sizeof(float) * n);
    cudaMalloc((void**) &real_gpu, sizeof(float) * n);
    cudaMalloc((void**) &image_gpu, sizeof(float) * n);
    cudaMemcpy(vector_gpu, vector, sizeof(float) * n, cudaMemcpyHostToDevice);

    cufftComplex *vector_complex, *fft_result;
    cudaMalloc((void**) &vector_complex, sizeof(cufftComplex) * n);
    cudaMalloc((void**) &fft_result, sizeof(cufftComplex) * n);
    dim3 grid(ceil(n / 256.));
    dim3 block(256);

    realToComplex<<<grid, block>>>(vector_gpu, vector_complex, n);
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    cufftExecC2C(plan, vector_complex, fft_result, CUFFT_FORWARD);
    complexToCPU<<<grid, block>>>(fft_result, real_gpu, image_gpu, n);
    cudaMemcpy(real, real_gpu, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(image, image_gpu, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(vector_gpu);
    cudaFree(real_gpu);
    cudaFree(image_gpu);
    cudaFree(vector_complex);
    cudaFree(fft_result);
}

__global__ void complexMul(cufftComplex* v1, cufftComplex* v2, cufftComplex* result, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n) {
        result[idx].x = (v1[idx].x * v2[idx].x - v1[idx].y * v2[idx].y) / n;
        result[idx].y = (v1[idx].x * v2[idx].y + v1[idx].y * v2[idx].x) / n;
    }
}

__global__ void complexToReal(cufftComplex* complex, float* result, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n) {
        result[idx] = complex[idx].x / n;
    }
}


void FFTCONV1D(float* vector , float* kernel, float* result, int k , int n) {
    float* vector_gpu, *kernel_gpu, *result_gpu;
    cudaMalloc((void**) &vector_gpu, sizeof(float) * n);
    cudaMalloc((void**) &kernel_gpu, sizeof(float) * n);
    cudaMalloc((void**) &result_gpu, sizeof(float) * n);

    cudaMemcpy(vector_gpu, vector, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemset(kernel_gpu, 0., sizeof(float) * n);
    cudaMemcpy(kernel_gpu, kernel, sizeof(float) * k, cudaMemcpyHostToDevice);

    cufftComplex *vector_fft;
    cudaMalloc((void**) &vector_fft, sizeof(cufftComplex) * n);

    cufftComplex *kernel_fft, *cov_result_gpu;
    cudaMalloc((void**) &kernel_fft, sizeof(cufftComplex) * n);
    cudaMalloc((void**) &cov_result_gpu, sizeof(cufftComplex) * n);

    dim3 grid(ceil(n / 256.));
    dim3 block(256);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_R2C, 1);
    cufftExecR2C(plan, vector_gpu, vector_fft);
    cufftExecR2C(plan, kernel_gpu, kernel_fft);



    complexMul<<<grid, block>>>(vector_fft, kernel_fft, cov_result_gpu, n);
    cufftHandle plan_I;
    cufftPlan1d(&plan_I, n, CUFFT_C2R, 1);
    cufftExecC2R(plan_I, cov_result_gpu, result_gpu);

    cudaMemcpy(result, result_gpu, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cufftDestroy(plan_I);
    cudaFree(vector_gpu);
    cudaFree(kernel_gpu);
    cudaFree(result_gpu);

    cudaFree(vector_fft);

    cudaFree(kernel_fft);
    cudaFree(cov_result_gpu);

}

__global__ void MatrixElementMult_kernel(float* m1, float* m2, float* result, int m, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < m && idy < n) {
        int index = idx * n + idy;
        result[index] = m1[index] * m2[index];
    }
}


extern "C"
void MatrixElementMult(float* m1, float* m2, float* result, int m, int n) {
    float *m1_gpu, *m2_gpu, *result_gpu;
    int size = sizeof(float) * m * n;
    cudaMalloc((void**) &m1_gpu, size);
    cudaMalloc((void**) &m2_gpu, size);
    cudaMalloc((void**) &result_gpu, size);
    cudaMemcpy(m1_gpu, m1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(m2_gpu, m2, size, cudaMemcpyHostToDevice);
    cudaMemset(result_gpu, 0., size);
    dim3 grid(ceil(m / 16.), ceil(n / 16.));
    dim3 block(16, 16);

#ifdef DEBUG
    printf("[GRID X] : %d, [BLOCK X] : %d\n", grid.x, block.x);
#endif

    MatrixElementMult_kernel<<<grid, block>>>(m1_gpu, m2_gpu, result_gpu, m, n);
    cudaDeviceSynchronize();
    cudaMemcpy(result, result_gpu, size, cudaMemcpyDeviceToHost);

    cudaFree(m1_gpu);
    cudaFree(m2_gpu);
    cudaFree(result_gpu);
}


__device__ __host__ void ComplexMultFunction(cufftComplex& input1, cufftComplex& input2, cufftComplex& output, int n, bool need_scale) {
    if(need_scale) {
        output.x = (input1.x * input2.x - input1.y * input2.y) / n;
        output.y = (input1.x * input2.y + input1.y * input2.x) / n;
    }else{
        output.x = input1.x * input2.x - input1.y * input2.y;
        output.y = input1.x * input2.y + input1.y * input2.x;
    }
}

__global__ void FFTCONV2D_kernel(cufftComplex* matrix, cufftComplex* kernel, cufftComplex* result, int m, int n, bool need_scale) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int index = idy * m + idx;

    if(idx < m && idy < n) {
        ComplexMultFunction(matrix[index], kernel[index], result[index], m * n, need_scale);
    }
    // printf("[IDX] : %d, [IDY] : %d", idx, idy);
}

extern "C"
void FFTCONV2D(float* m1, float* m2 , float* result, int m, int n, int k) {
    bool scale = true;
    assert(k < m && k < n);
    float *m1_gpu, *m2_gpu, *result_gpu;
    int size = sizeof(float) * m * n;
    cudaMallocManaged((void **) &m1_gpu, size);
    cudaMallocManaged((void **) &m2_gpu, size);
    cudaMallocManaged((void **) &result_gpu, size);

    cudaMemset(m2_gpu, 0, size);
    for(int i = 0; i < k; ++i) {
        for(int j = 0; j < k; ++j) {
            m2_gpu[i * n + j] = m2[i * k + j];
        }
    }

    cudaMemcpy(m1_gpu, m1, size, cudaMemcpyHostToDevice);

    int size_fft = sizeof(cufftComplex) * m * n; 
    cufftComplex *m1_fft, *m2_fft, *result_fft;
    cudaMallocManaged((void **) &m1_fft, size_fft);
    cudaMallocManaged((void **) &m2_fft, size_fft);
    cudaMallocManaged((void **) &result_fft, size_fft);

    cufftHandle fftPlan;
    cufftPlan2d(&fftPlan, m, n, CUFFT_R2C);
    cufftExecR2C(fftPlan, m1_gpu, m1_fft);
    cufftExecR2C(fftPlan, m2_gpu, m2_fft);

    cudaError_t err = cudaGetLastError();
    

    cudaFree(m1_gpu);
    cudaFree(m2_gpu);
    cufftDestroy(fftPlan);

    dim3 grid(ceil(m / 32.), ceil(n / 32.));
    dim3 block(32, 32);

#ifdef DEBUG
    printf("[GRID X] : %d, [BLOCK X] : %d, [GRID Y] : %d, [BLOCK Y] : %d\n",grid.x, block.x, grid.y, block.y);
#endif

    FFTCONV2D_kernel<<<grid, block>>>(m1_fft, m2_fft, result_fft, m, n, scale);
    
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        viewCudaDeviceInfo();
        exit(-1);
    } 
    
    cudaDeviceSynchronize();
    
    cudaFree(m1_fft);
    cudaFree(m2_fft);

    cufftHandle ifftPlan;
    cufftPlan2d(&ifftPlan, m, n, CUFFT_C2R);
    cufftExecC2R(ifftPlan, result_fft, result_gpu);
    cudaFree(result_fft);
    cufftDestroy(ifftPlan);

    cudaMemcpy(result, result_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree(result_gpu);
}





