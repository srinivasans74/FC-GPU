#include <hip/hip_runtime.h> // Required for HIP API calls
#include <iostream>          // For standard input/output
#include <vector>            // For std::vector (optional, but good practice for host data)

// Macro for HIP error checking
#define HIP_CHECK(call)                                                          \
    do {                                                                         \
        hipError_t err = call;                                                   \
        if (err != hipSuccess) {                                                 \
            std::cerr << "HIP Error: " << hipGetErrorString(err) << " at "       \
                      << __FILE__ << ":" << __LINE__ << " in " << #call << std::endl; \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

/**
 * @brief HIP kernel for matrix multiplication C = A * B.
 * Each thread computes one element of the resulting matrix C.
 * @param A Pointer to the input matrix A on the device.
 * @param B Pointer to the input matrix B on the device.
 * @param C Pointer to the output matrix C on the device.
 * @param N Dimension of the square matrices (N x N).
 */
__global__ void matMulKernel(float* A, float* B, float* C, int N) {
    // Calculate global row and column index for the current thread
    // blockIdx.y * blockDim.y + threadIdx.y gives the global row index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // blockIdx.x * blockDim.x + threadIdx.x gives the global column index
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current thread is within the matrix bounds
    if (row < N && col < N) {
        float val = 0.0f;
        // Perform dot product for the C[row][col] element
        // C[row][col] = sum(A[row][k] * B[k][col]) for k from 0 to N-1
        for (int k = 0; k < N; ++k) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

int main() {
    // Define matrix dimension
    const int N = 256; // Using a smaller N for a quick example
    const int SIZE = N * N; // Total number of elements in an N x N matrix

    // Host-side pointers for pinned and mapped memory
    float *h_A, *h_B, *h_C;
    // Device-side pointers (aliases to host-mapped memory)
    float *d_A, *d_B, *d_C;

    // 1. Enable host memory mapping
    // This flag allows the GPU to directly access host-allocated memory.
    HIP_CHECK(hipSetDeviceFlags(hipDeviceMapHost));

    // 2. Allocate page-locked (pinned) and host-mapped memory
    // The (void**) cast is crucial here as hipHostAlloc expects void**
    HIP_CHECK(hipHostAlloc((void**)&h_A, SIZE * sizeof(float), hipHostMallocMapped));
    HIP_CHECK(hipHostAlloc((void**)&h_B, SIZE * sizeof(float), hipHostMallocMapped));
    HIP_CHECK(hipHostAlloc((void**)&h_C, SIZE * sizeof(float), hipHostMallocMapped));

    // 3. Get device pointers (aliases) for the host-mapped memory
    // These pointers can be passed directly to the kernel.
    // The (void**) cast is also needed for hipHostGetDevicePointer.
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_A, h_A, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_B, h_B, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_C, h_C, 0));

    // 4. Initialize host-mapped memory on the CPU
    // No hipMemcpy is needed because the memory is host-mapped;
    // the device can directly read from and write to these host addresses.
    for (int i = 0; i < SIZE; ++i) {
        h_A[i] = static_cast<float>(i % 10); // Simple initialization
        h_B[i] = static_cast<float>((i % 5) + 1); // Avoid zeros for B
        h_C[i] = 0.0f; // Initialize C to zero
    }

    // 5. Define grid and block dimensions for the kernel launch
    // A common block size is 16x16 threads.
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    // Calculate grid dimensions to cover the entire N x N matrix
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // 6. Launch the kernel
    std::cout << "Launching matMulKernel with N=" << N << "..." << std::endl;
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    HIP_CHECK(hipGetLastError()); // Check for any errors during kernel launch

    // 7. Synchronize the device to ensure kernel completion
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "Kernel execution complete." << std::endl;

    // 8. Verify results (optional, but good for testing)
    // Since h_C is host-mapped, the results are directly available on the host.
    bool success = true;
    // Simple check for a few elements
    if (N >= 2) {
        // Expected C[0][0] = sum(A[0][k] * B[k][0])
        float expected_C00 = 0.0f;
        for (int k = 0; k < N; ++k) {
            expected_C00 += (static_cast<float>(k % 10)) * (static_cast<float>((k % 5) + 1));
        }
        if (std::abs(h_C[0] - expected_C00) > 0.001f) {
            std::cerr << "Verification failed for C[0][0]: Expected " << expected_C00
                      << ", got " << h_C[0] << std::endl;
            success = false;
        } else {
            std::cout << "Verification successful for C[0][0]: " << h_C[0] << std::endl;
        }
    }

    // 9. Free allocated memory
    // hipHostFree releases both host and device views for mapped memory.
    HIP_CHECK(hipHostFree(h_A));
    HIP_CHECK(hipHostFree(h_B));
    HIP_CHECK(hipHostFree(h_C));

    if (success) {
        std::cout << "Program finished successfully." << std::endl;
    } else {
        std::cerr << "Program finished with verification errors." << std::endl;
        return 1;
    }

    return 0;
}
