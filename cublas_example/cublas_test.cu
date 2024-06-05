#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>


// Define some error checking macros.
#define cudaErrCheck(stat)                       \
   {                                             \
      cudaErrCheck_((stat), __FILE__, __LINE__); \
   }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
   if (stat != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

int8_t float2int8(float f, float scale) {
    int8_t i = int8_t(f * scale);
    if (i < -127) i = -127;
    if (i > 127) i = 127;
    return i;
}

template <typename T, typename S>
void allocate_memory(int m, int n, int k, T **A, T **B, S **C) {
    cudaMallocManaged(A, m * k * sizeof(T));
    cudaMallocManaged(B, k * n * sizeof(T));
    cudaMallocManaged(C, m * n * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

template <typename T, typename S>
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                   int m, int n, int k, T *A, T *B, S *C, int lda, int ldb, int ldc,
                   S *alpha, S *beta, int algo, int id) {
    cudaDataType_t AType, BType, CType, ComputeType;
    if (id==0) {
        AType = BType = CType = ComputeType = CUDA_R_32F;
    } else if (id==1) {
        AType = BType = CUDA_R_16F;
        ComputeType = CUDA_R_32F;
        CType =CUDA_R_32F;
    } else if (id ==2) {
        AType = BType = CUDA_R_8I;
        CType = ComputeType = CUDA_R_32I;
    } else {
        // printf("Not supported data type.");
        return -1;
    }
    cublasStatus_t status;
    status = cublasGemmEx(handle,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          alpha,
                          A,
                          AType,
                          lda,
                          B,
                          BType,
                          ldb,
                          beta,
                          C,
                          CType,
                          ldc,
                          ComputeType,
                          static_cast<cublasGemmAlgo_t>(algo));

    if (status == CUBLAS_STATUS_SUCCESS)
        return 1;
    else
        return -1;
}

template <typename T, typename S>
// void test_gemm(cublasHandle_t handle, int m, int n, int k, T *A, T *B, S *C,
//                S *alpha, S *beta, int algo, int iteration, int id) {
void test_gemm(cublasHandle_t handle, int m, int n, int k, T *A, T *B, S *C,
               S *alpha, S *beta, int algo, int iteration, int id) {
    float total_time = 0;
    for (int i = 0; i < iteration; ++i) {
        struct timeval start, end;
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);
        int success = cublas_gemm_ex(handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     n,
                                     m,
                                     k,
                                     B,
                                     A,
                                     C,
                                     n,
                                     k,
                                     n,
                                     alpha,
                                     beta,
                                     static_cast<cublasGemmAlgo_t>(algo),id);
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        cudaProfilerStop();
        if (success > 0 && i > 0)
            total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
    if (total_time > 0)
        printf("algo %d: %.3f ms\n", algo, total_time / (iteration - 1));
}

int main() {
    // int m = 4096, n = 8192, k = 1024;
    // int m = 4096, n = 4096, k = 4096;
    // int m = 128, n = 256, k = 64;
    int m = 256, n = 256, k = 256;
    printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);
    int start_algo = CUBLAS_GEMM_DEFAULT;
    int end_algo = CUBLAS_GEMM_ALGO23;
    int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;
    int iteration = 1;

    float *fA, *fB, *fC, *fA2, *fB2, *fC2;
    __half *hA, *hB, *hA2, *hB2;
    float *hC, *hC2;
    // int8_t *iA, *iB; int32_t *iC;
    float f_alpha = 1, f_beta = 0;
    __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
    // int32_t i_alpha = 1, i_beta = 0;

    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cudaErrCheck(cudaGetDevice(&cuda_device));

    cudaErrCheck(cudaGetDeviceProperties(&deviceProp, cuda_device));

    if ((deviceProp.concurrentKernels == 0)) {
        //printf("> GPU does not support concurrent kernel execution\n");
        //printf("  CUDA kernel runs will be serialized\n");
    } else {
        //printf("concurrent kernel: %d\n",deviceProp.concurrentKernels);
    }

    allocate_memory(m, n, k, &fA, &fB, &fC);
    allocate_memory(m, n, k, &hA, &hB, &hC);
    allocate_memory(m, n, k, &fA2, &fB2, &fC2);
    allocate_memory(m, n, k, &hA2, &hB2, &hC2);
    // allocate_memory(m, n, k, &iA, &iB, &iC);
    for (int i = 0; i < m * k; ++i) {
        fA[i] = float(i % 255 - 127) / 127;
        hA[i] = __float2half_rn(fA[i]);
        // iA[i] = float2int8(fA[i], 127);
    } 
    for (int i = 0; i < k * n; ++i) {
        fB[i] = float(i % 255 - 127) / 127;
        hB[i] = __float2half_rn(fB[i]);
        // iB[i] = float2int8(fB[i], 127);
    }
    for (int i = 0; i < m * k; ++i) {
        fA2[i] = float(i % 255 - 127) / 127;
        hA2[i] = __float2half_rn(fA2[i]);
        // iA[i] = float2int8(fA[i], 127);
    } 
    for (int i = 0; i < k * n; ++i) {
        fB2[i] = float(i % 255 - 127) / 127;
        hB2[i] = __float2half_rn(fB2[i]);
        // iB[i] = float2int8(fB[i], 127);
    }

    cudaStream_t stream1, stream2;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
    //!sp
    // for (int algo = start_algo; algo <= end_algo; ++algo)
    //     test_gemm(handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration,0);
    // for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
    //     test_gemm(handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration,0);
    

    printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");

    // cublasSetStream(handle, stream1);
    test_gemm(handle, m, n, k, hA, hB, hC, &f_alpha, &f_beta, 4, iteration, 1);
    // cublasSetStream(handle, stream2);
    test_gemm(handle, m, n, k, hA2, hB2, hC2, &f_alpha, &f_beta, 110, iteration, 1);

    //! sp
    // for (int algo = 0; algo <= end_algo; ++algo)
    //     test_gemm(handle, m, n, k, hA, hB, hC, &f_alpha, &f_beta, algo, iteration,1);
    //! tc
    // for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
    //     test_gemm(handle, m, n, k, hA, hB, hC, &f_alpha, &f_beta, algo, iteration,1);
        

    // printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
    // for (int algo = start_algo; algo <= end_algo; ++algo)
    //     test_gemm(handle, m, n, k, iA, iB, iC, &i_alpha, &i_beta, algo, iteration,2);
    // for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
    //     test_gemm(handle, m, n, k, iA, iB, iC, &i_alpha, &i_beta, algo, iteration,2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
    // printf("fp32: ");
    // for (int i = 0; i < 10; ++i)
    //     printf("%.5f%c", fC[i], " \n"[i==9]);
    printf("fp16: ");
    for (int i = 0; i < 10; ++i)
        printf("%.5f%c", float(hC[i]), " \n"[i==9]);
    printf("tensor: ");
    for (int i = 0; i < 10; ++i)
        printf("%.5f%c", float(hC2[i]), " \n"[i==9]);
    // printf("int8: ");
    // for (int i = 0; i < 10; ++i)
    //     printf("%.5f%c", float(iC[i])/127/127, " \n"[i==9]);

    cublasDestroy(handle);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // free_memory(iA, iB, iC);
    free_memory(fA, fB, fC);
    free_memory(hA, hB, hC);
    free_memory(fA2, fB2, fC2);
    free_memory(hA2, hB2, hC2);
    return 0;
}