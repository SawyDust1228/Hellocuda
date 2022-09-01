
#include "kernel.cuh"

#define DEBUG

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
    cudaMemcpy(m_gpu, m, size_mask, cudaMemcpyHostToDevice);


    dim3 grid(ceil(n / 256.));
    dim3 block(256);
#ifdef DEBUG
    printf("[GRID X] : %d, [BLOCK X] : %d\n", grid.x, block.x);
#endif
    conv1d_kernel<<<grid, block>>>(v_gpu, result_gpu, m_gpu, n, k);

    cudaMemcpy(result, result_gpu, size, cudaMemcpyDeviceToHost);
    
    cudaFree(v_gpu);
    cudaFree(result_gpu);
    cudaFree(m_gpu);
}