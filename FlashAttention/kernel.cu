#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
Key Principles:
Thread Utilization:Ensure that the number of threads per block is a multiple of warpSize (typically 32) to avoid underutilization of hardware resources.
Shared Memory Alignment:When using shared memory, BLOCK_SIZE or TILE_WIDTH should divide evenly into NUM_SAMPLES and FEATURE_DIMENSION to minimize boundary conditions and wasted computations.
Grid Size Optimization:Ensure the grid dimensions align with the block size so that each thread block processes a uniform amount of work.
Avoid Divergence:Avoid irregular boundary conditions by padding NUM_SAMPLES or FEATURE_DIMENSION to the nearest multiple of BLOCK_SIZE if they are not already divisible.
These adjustments will improve efficiency by minimizing divergence and maximizing thread utilization and shared memory bandwidth.
*/

#define BLOCK_SIZE 32
#define MEM_WIDTH 32
#define TILE_WIDTH 32

#define NUM_SAMPLES 1024
#define FEATURE_DIMENSION 1024

// Utility macro for CUDA error checking
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, const char* const func, const char* const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line
            << " '" << func << "'\n";
        std::cerr << "Error: " << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

// Matrix Initialization
void generateRandomMatrix(float* matrix, int rows, int cols) {
    srand(0);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Matrix Printing
void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            printf("%.6f ", matrix[y * cols + x]);
        }
        printf("\n");
    }
    printf("\n");
}

// Comparison of Matrices
void compareMatrices(const float* cpuMatrix, const float* gpuMatrix, int rows, int cols, const char* name) {
    const float epsilon = 1e-5f;
    bool match = true;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int index = y * cols + x;
            float diff = fabs(cpuMatrix[index] - gpuMatrix[index]);
            if (diff > epsilon) {
                printf("Mismatch in %s at [%d][%d]: CPU=%.6f, GPU=%.6f, Diff=%.6f\n",
                    name, y, x, cpuMatrix[index], gpuMatrix[index], diff);
                match = false;
            }
        }
    }

    if (match) {
        printf("Success: %s matrices match within tolerance %.6f\n", name, epsilon);
    }
}

// Transpose Matrix
void transposeMatrix(const float* inputMatrix, float* transposedMatrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposedMatrix[j * rows + i] = inputMatrix[i * cols + j];
        }
    }
}

// CPU Implementation of Attention
void computeAttentionCPU(float* query, float* key, float* value, float* attentionScores, float* output) {
    float* transposedKey = (float*)malloc(FEATURE_DIMENSION * NUM_SAMPLES * sizeof(float));
    transposeMatrix(key, transposedKey, NUM_SAMPLES, FEATURE_DIMENSION);

    float scalingFactor = 1.0f / sqrtf((float)FEATURE_DIMENSION);

    // Compute attention scores
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_SAMPLES; j++) {
            for (int k = 0; k < FEATURE_DIMENSION; k++) {
                attentionScores[i * NUM_SAMPLES + j] += query[i * FEATURE_DIMENSION + k] * transposedKey[k * NUM_SAMPLES + j];
            }
            attentionScores[i * NUM_SAMPLES + j] *= scalingFactor;
        }
    }

    // Apply softmax row-wise
    for (int row = 0; row < NUM_SAMPLES; row++) {
        float maxScore = attentionScores[row * NUM_SAMPLES];
        for (int col = 1; col < NUM_SAMPLES; col++) {
            if (attentionScores[row * NUM_SAMPLES + col] > maxScore) {
                maxScore = attentionScores[row * NUM_SAMPLES + col];
            }
        }

        float sumExp = 0.0f;
        for (int col = 0; col < NUM_SAMPLES; col++) {
            attentionScores[row * NUM_SAMPLES + col] = exp(attentionScores[row * NUM_SAMPLES + col] - maxScore);
            sumExp += attentionScores[row * NUM_SAMPLES + col];
        }

        for (int col = 0; col < NUM_SAMPLES; col++) {
            attentionScores[row * NUM_SAMPLES + col] /= sumExp;
        }
    }

    // Compute output = attentionScores * value
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < FEATURE_DIMENSION; j++) {
            for (int k = 0; k < NUM_SAMPLES; k++) {
                output[i * FEATURE_DIMENSION + j] += attentionScores[i * NUM_SAMPLES + k] * value[k * FEATURE_DIMENSION + j];
            }
        }
    }

    free(transposedKey);
}


// ---------------------------------- GPU_Global ----------------------------------
// CUDA kernel for scaled dot-product QK^T
__global__ void computeScoresKernel(float* queryMatrix, float* keyMatrix, float* scoreMatrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in the matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index in the matrix

    if (row < NUM_SAMPLES && col < NUM_SAMPLES) {
        float score = 0.0f;
        for (int d = 0; d < FEATURE_DIMENSION; ++d) {
            score += queryMatrix[row * FEATURE_DIMENSION + d] * keyMatrix[col * FEATURE_DIMENSION + d];
        }
        scoreMatrix[row * NUM_SAMPLES + col] = score / sqrtf(static_cast<float>(FEATURE_DIMENSION)); // Scale
    }
}

// CUDA kernel for applying softmax to rows
__global__ void applySoftmaxKernel(float* scoreMatrix, float* softmaxMatrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (row < NUM_SAMPLES) {
        float maxScore = -1e30f;

        // Find max score for numerical stability
        for (int col = 0; col < NUM_SAMPLES; ++col) {
            maxScore = fmaxf(maxScore, scoreMatrix[row * NUM_SAMPLES + col]);
        }

        float sumExp = 0.0f;

        // Compute exponentials
        for (int col = 0; col < NUM_SAMPLES; ++col) {
            softmaxMatrix[row * NUM_SAMPLES + col] = expf(scoreMatrix[row * NUM_SAMPLES + col] - maxScore);
            sumExp += softmaxMatrix[row * NUM_SAMPLES + col];
        }

        // Normalize
        for (int col = 0; col < NUM_SAMPLES; ++col) {
            softmaxMatrix[row * NUM_SAMPLES + col] /= sumExp;
        }
    }
}

// CUDA kernel for computing final output matrix = softmax_scores * V
__global__ void computeOutputKernel(float* softmaxMatrix, float* valueMatrix, float* outputMatrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in the output
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index in the output

    if (row < NUM_SAMPLES && col < FEATURE_DIMENSION) {
        float result = 0.0f;
        for (int k = 0; k < NUM_SAMPLES; ++k) {
            result += softmaxMatrix[row * NUM_SAMPLES + k] * valueMatrix[k * FEATURE_DIMENSION + col];
        }
        outputMatrix[row * FEATURE_DIMENSION + col] = result;
    }
}

// GPU-based implementation of Flash Attention (Global Memory)
void computeAttentionGPUGlobal(float* queryMatrix, float* keyMatrix, float* valueMatrix, float* attentionMatrix, float* outputMatrix) {
    // Device pointers
    float* d_queryMatrix, * d_keyMatrix, * d_valueMatrix;
    float* d_scoreMatrix, * d_softmaxMatrix, * d_outputMatrix;

    // Allocate device memory
    cudaMalloc(&d_queryMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&d_keyMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&d_valueMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&d_scoreMatrix, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&d_softmaxMatrix, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&d_outputMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_queryMatrix, queryMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keyMatrix, keyMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valueMatrix, valueMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);

    // Configure thread block and grid dimensions
    // Use BLOCK_SIZE for the block dimensions
    // Compute grid dimensions to cover NUM_SAMPLES
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads per block
    dim3 gridDim((NUM_SAMPLES + blockDim.x - 1) / blockDim.x, (NUM_SAMPLES + blockDim.y - 1) / blockDim.y);

    // Compute QK^T
    computeScoresKernel << <gridDim, blockDim >> > (d_queryMatrix, d_keyMatrix, d_scoreMatrix);
    cudaDeviceSynchronize();

    // Apply softmax to scores
    dim3 softmaxBlockDim(1, 256);
    dim3 softmaxGridDim(1, (NUM_SAMPLES + softmaxBlockDim.y - 1) / softmaxBlockDim.y);
    applySoftmaxKernel << <softmaxGridDim, softmaxBlockDim >> > (d_scoreMatrix, d_softmaxMatrix);
    cudaDeviceSynchronize();

    // Compute final output
    dim3 outputBlock(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads for output matrix
    dim3 outputGrid((FEATURE_DIMENSION + outputBlock.x - 1) / outputBlock.x, (NUM_SAMPLES + outputBlock.y - 1) / outputBlock.y);
    computeOutputKernel << <outputGrid, outputBlock >> > (d_softmaxMatrix, d_valueMatrix, d_outputMatrix);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(attentionMatrix, d_softmaxMatrix, NUM_SAMPLES * NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputMatrix, d_outputMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_queryMatrix);
    cudaFree(d_keyMatrix);
    cudaFree(d_valueMatrix);
    cudaFree(d_scoreMatrix);
    cudaFree(d_softmaxMatrix);
    cudaFree(d_outputMatrix);
}


// -------------------------------------------------------------------------------------------------------------

// ---------------------------------- GPU_Shared ----------------------------------

// CUDA kernel for scaled dot-product QK^T
__global__ void shared_compute_scores(float* queryMatrix, float* keyTransposeMatrix, float* attentionScores) {

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int scoreColumnIndex = blockX * TILE_WIDTH + threadX;
    int scoreRowIndex = blockY * TILE_WIDTH + threadY;
    float scoreValue = 0.0f;

    // Determine the number of phases
    int numPhases = (FEATURE_DIMENSION + TILE_WIDTH - 1) / TILE_WIDTH;

    // Initialize shared memory
    __shared__ float sharedQuery[MEM_WIDTH][MEM_WIDTH];
    __shared__ float sharedKeyTranspose[MEM_WIDTH][MEM_WIDTH];

    // Iterate through phases
    for (int phase = 0; phase < numPhases; phase++) {
        // boundary check if the elements should be 
        // loaded into shared mem.

        // The element must be valid in Q or K_T. Checking if within row and col range
        // NOTICE! Loading is only determined by if there are valid elements inside M or N!
        // The tile might be out of bound for output P, but there is still valid element
        // inside M or N that needs to be loaded and used by others!
        if (phase * TILE_WIDTH + threadX < FEATURE_DIMENSION && blockY * TILE_WIDTH + threadY < NUM_SAMPLES) {
            sharedQuery[threadY][threadX] = queryMatrix[(blockY * TILE_WIDTH + threadY) * FEATURE_DIMENSION + phase * TILE_WIDTH + threadX];
        }
        else {
            sharedQuery[threadY][threadX] = 0.0f;
        }

        if (phase * TILE_WIDTH + threadY < FEATURE_DIMENSION && blockX * TILE_WIDTH + threadX < NUM_SAMPLES) {
            sharedKeyTranspose[threadY][threadX] = keyTransposeMatrix[(phase * TILE_WIDTH + threadY) * NUM_SAMPLES + blockX * TILE_WIDTH + threadX];
        }
        else {
            sharedKeyTranspose[threadY][threadX] = 0.0f;
        }
        // synchronize to ensure all the loadings finish
        __syncthreads();

        // Accumulate dot product for the current tile
        if (scoreColumnIndex < NUM_SAMPLES && scoreRowIndex < NUM_SAMPLES) {
            // Cumulatively add the scores_value based on elements in the tile
            for (int i = 0; i < TILE_WIDTH; i++) {
                scoreValue += sharedQuery[threadY][i] * sharedKeyTranspose[i][threadX];
            }
        }

        __syncthreads();
    }
    // Add the scores_value to the output scores with phases if it has a valid coordinate
    // Write the computed score to the global memory
    if (scoreColumnIndex < NUM_SAMPLES && scoreRowIndex < NUM_SAMPLES) {
        attentionScores[scoreRowIndex * NUM_SAMPLES + scoreColumnIndex] = scoreValue / sqrtf(static_cast<float>(FEATURE_DIMENSION));
    }
}

// CUDA kernel for applying softmax
__global__ void shared_softmax(float* attentionScores, float* softmaxScores) {
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowIndex < NUM_SAMPLES) {
        float maxScore = -1e30f;

        // Find max score for numerical stability
        for (int colIndex = 0; colIndex < NUM_SAMPLES; ++colIndex) {
            maxScore = fmaxf(maxScore, attentionScores[rowIndex * NUM_SAMPLES + colIndex]);
        }

        float sumExp = 0.0f;

        // Compute exponentials
        for (int colIndex = 0; colIndex < NUM_SAMPLES; ++colIndex) {
            softmaxScores[rowIndex * NUM_SAMPLES + colIndex] = expf(attentionScores[rowIndex * NUM_SAMPLES + colIndex] - maxScore);
            sumExp += softmaxScores[rowIndex * NUM_SAMPLES + colIndex];
        }

        // Normalize
        for (int colIndex = 0; colIndex < NUM_SAMPLES; ++colIndex) {
            softmaxScores[rowIndex * NUM_SAMPLES + colIndex] /= sumExp;
        }
    }
}

//// CUDA kernel for applying softmax
//__global__ void shared_softmax(float* attentionScores, float* softmaxScores) {
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//
//    int Row = blockIdx.y;
//    int Col = tx;
//
//    __shared__ float ds_scores[NUM_SAMPLES];
//
//    if (Row < NUM_SAMPLES && Col < NUM_SAMPLES) {
//        ds_scores[Col] = attentionScores[Row * NUM_SAMPLES + Col];
//    }
//    else {
//        ds_scores[Col] = -1e30f;
//    }
//    __syncthreads();
//
//    if (Row < NUM_SAMPLES && Col < NUM_SAMPLES) {
//        float max_val = -1e10f;
//        for (int i = 0; i < NUM_SAMPLES; i++) {
//            max_val = fmax(max_val, ds_scores[i]);
//        }
//
//        ds_scores[Col] = expf(ds_scores[Col] - max_val);
//    }
//    __syncthreads();
//
//    if (Row < NUM_SAMPLES && Col < NUM_SAMPLES) {
//        float exp_sum = 0.0f;
//        for (int j = 0; j < NUM_SAMPLES; j++) {
//            exp_sum += ds_scores[j];
//        }
//        softmaxScores[Row * NUM_SAMPLES + Col] = ds_scores[Col] / exp_sum;
//    }
//}

// CUDA kernel for computing output = softmax_scores * V
__global__ void shared_compute_output(float* softmaxScores, float* valueMatrix, float* outputMatrix) {

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int outputColumnIndex = blockX * TILE_WIDTH + threadX;
    int outputRowIndex = blockY * TILE_WIDTH + threadY;
    float outputValue = 0.0f;

    // Determine the number of phases
    int numPhases = (NUM_SAMPLES + TILE_WIDTH - 1) / TILE_WIDTH;

    // Initialize shared memory
    __shared__ float sharedSoftmaxScores[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedValueMatrix[TILE_WIDTH][TILE_WIDTH];

    // Iterate through phases
    for (int phase = 0; phase < numPhases; phase++) {
        // Load valid elements into shared memory
        if (phase * TILE_WIDTH + threadX < NUM_SAMPLES && blockY * TILE_WIDTH + threadY < NUM_SAMPLES) {
            sharedSoftmaxScores[threadY][threadX] = softmaxScores[(blockY * TILE_WIDTH + threadY) * NUM_SAMPLES + phase * TILE_WIDTH + threadX];
        }
        else {
            sharedSoftmaxScores[threadY][threadX] = 0.0f;
        }

        if (phase * TILE_WIDTH + threadY < NUM_SAMPLES && blockX * TILE_WIDTH + threadX < FEATURE_DIMENSION) {
            sharedValueMatrix[threadY][threadX] = valueMatrix[(phase * TILE_WIDTH + threadY) * FEATURE_DIMENSION + blockX * TILE_WIDTH + threadX];
        }
        else {
            sharedValueMatrix[threadY][threadX] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for the output matrix
        if (outputColumnIndex < FEATURE_DIMENSION && outputRowIndex < NUM_SAMPLES) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                outputValue += sharedSoftmaxScores[threadY][i] * sharedValueMatrix[i][threadX];
            }
        }

        // synchronize for phases
        __syncthreads();
    }

    // Write the computed output to the global memory
    if (outputColumnIndex < FEATURE_DIMENSION && outputRowIndex < NUM_SAMPLES) {
        outputMatrix[outputRowIndex * FEATURE_DIMENSION + outputColumnIndex] = outputValue;
    }
}

// GPU Implementation of Flash Attention
void computeAttentionGPUShared(float* queryMatrix, float* keyTransposeMatrix, float* valueMatrix, float* attentionScores, float* outputMatrix) {
    float* deviceQuery, * deviceKeyTranspose, * deviceValue, * deviceAttentionScores, * deviceSoftmaxScores, * deviceOutput;

    // Allocate device memory
    cudaMalloc(&deviceQuery, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceKeyTranspose, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceValue, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));
    cudaMalloc(&deviceAttentionScores, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&deviceSoftmaxScores, NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
    cudaMalloc(&deviceOutput, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float));

    // Copy data to device
    cudaMemcpy(deviceQuery, queryMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKeyTranspose, keyTransposeMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValue, valueMatrix, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    // Use TILE_WIDTH for blockDimension
    // Compute grid dimensions based on TILE_WIDTH
    dim3 blockDimension(TILE_WIDTH, TILE_WIDTH); // 32*32 threads per block
    dim3 gridDimension((NUM_SAMPLES + blockDimension.x - 1) / blockDimension.x, (NUM_SAMPLES + blockDimension.y - 1) / blockDimension.y);

    // Launch kernels
    shared_compute_scores << <gridDimension, blockDimension >> > (deviceQuery, deviceKeyTranspose, deviceAttentionScores);
    cudaDeviceSynchronize();

    // Set softmaxBlockDim to be BLOCK_SIZE or a divisor of NUM_SAMPLES
    // Adjust softmaxGridDim accordingly
    dim3 softmaxBlockDimension(1, BLOCK_SIZE); // Threads for softmax rows
    dim3 softmaxGridDimension(1, (NUM_SAMPLES + softmaxBlockDimension.y - 1) / softmaxBlockDimension.y);
    // dim3 softmaxBlockDimension(NUM_SAMPLES, 1); // Threads for softmax rows
    // dim3 softmaxGridDimension(1, NUM_SAMPLES);

    shared_softmax << <softmaxGridDimension, softmaxBlockDimension >> > (deviceAttentionScores, deviceSoftmaxScores);
    cudaDeviceSynchronize();

    // Use TILE_WIDTH for blockDimension
    // Compute grid dimensions based on TILE_WIDTH
    dim3 outputBlock(TILE_WIDTH, TILE_WIDTH); // 32*32 threads for output matrix
    dim3 outputGrid((FEATURE_DIMENSION + outputBlock.x - 1) / outputBlock.x, (NUM_SAMPLES + outputBlock.y - 1) / outputBlock.y);
    shared_compute_output << <outputGrid, outputBlock >> > (deviceSoftmaxScores, deviceValue, deviceOutput);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(attentionScores, deviceSoftmaxScores, NUM_SAMPLES * NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputMatrix, deviceOutput, NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceQuery);
    cudaFree(deviceKeyTranspose);
    cudaFree(deviceValue);
    cudaFree(deviceAttentionScores);
    cudaFree(deviceSoftmaxScores);
    cudaFree(deviceOutput);
}

// -------------------------------------------------------------------------------------------------------------

int main() {
    float* queryMatrix = new float[NUM_SAMPLES * FEATURE_DIMENSION];
    float* keyMatrix = new float[NUM_SAMPLES * FEATURE_DIMENSION];
    float* valueMatrix = new float[NUM_SAMPLES * FEATURE_DIMENSION];
    float* outputCPU = new float[NUM_SAMPLES * FEATURE_DIMENSION]();
    float* outputGPUGlobal = new float[NUM_SAMPLES * FEATURE_DIMENSION]();
    float* outputGPUShared = new float[NUM_SAMPLES * FEATURE_DIMENSION]();
    float* attentionScoresCPU = new float[NUM_SAMPLES * NUM_SAMPLES]();
    float* attentionScoresGlobal = new float[NUM_SAMPLES * NUM_SAMPLES]();
    float* attentionScoresShared = new float[NUM_SAMPLES * NUM_SAMPLES]();
    float* transposedKeyMatrix = new float[FEATURE_DIMENSION * NUM_SAMPLES];

    generateRandomMatrix(queryMatrix, NUM_SAMPLES, FEATURE_DIMENSION);
    generateRandomMatrix(keyMatrix, NUM_SAMPLES, FEATURE_DIMENSION);
    generateRandomMatrix(valueMatrix, NUM_SAMPLES, FEATURE_DIMENSION);
    transposeMatrix(keyMatrix, transposedKeyMatrix, NUM_SAMPLES, FEATURE_DIMENSION);

    /*printMatrix(queryMatrix, NUM_SAMPLES, FEATURE_DIMENSION, "Query Matrix");
    printMatrix(keyMatrix, NUM_SAMPLES, FEATURE_DIMENSION, "Key Matrix");
    printMatrix(valueMatrix, NUM_SAMPLES, FEATURE_DIMENSION, "Value Matrix");*/

    // Time CPU
    cudaEvent_t start, stop;
    float cpu_milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    computeAttentionCPU(queryMatrix, keyMatrix, valueMatrix, attentionScoresCPU, outputCPU);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_milliseconds, start, stop);

    // Print CPU Matrices
    //printMatrix(attentionScoresCPU, NUM_SAMPLES, NUM_SAMPLES, "CPU Attention Scores");
    //printMatrix(outputCPU, NUM_SAMPLES, FEATURE_DIMENSION, "CPU Output Matrix");

    // Time GPU Global
    float gpu_global_milliseconds = 0;
    cudaEventRecord(start);
    computeAttentionGPUGlobal(queryMatrix, keyMatrix, valueMatrix, attentionScoresGlobal, outputGPUGlobal);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_global_milliseconds, start, stop);
    //printMatrix(attentionScoresGlobal, NUM_SAMPLES, NUM_SAMPLES, "GPU Global Attention Scores");
    //printMatrix(outputGPUGlobal, NUM_SAMPLES, FEATURE_DIMENSION, "GPU Global Output Matrix");

    // Time GPU Shared
    float gpu_shared_milliseconds = 0;
    cudaEventRecord(start);
    computeAttentionGPUShared(queryMatrix, transposedKeyMatrix, valueMatrix, attentionScoresShared, outputGPUShared);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_shared_milliseconds, start, stop);
    //printMatrix(attentionScoresShared, NUM_SAMPLES, NUM_SAMPLES, "GPU Shared Attention Scores");
    //printMatrix(outputGPUShared, NUM_SAMPLES, FEATURE_DIMENSION, "GPU Shared Output Matrix");

    // Compare Outputs
    compareMatrices(attentionScoresCPU, attentionScoresGlobal, NUM_SAMPLES, NUM_SAMPLES, "GPU Global Memory vs CPU Attention Map");
    compareMatrices(attentionScoresCPU, attentionScoresShared, NUM_SAMPLES, NUM_SAMPLES, "GPU Shared Memory vs CPU Attention Map");
    compareMatrices(outputCPU, outputGPUGlobal, NUM_SAMPLES, FEATURE_DIMENSION, "GPU Global Memory vs CPU Output");
    compareMatrices(outputCPU, outputGPUShared, NUM_SAMPLES, FEATURE_DIMENSION, "GPU Shared Memory vs CPU Output");

    //printf("\nCPU Execution Time: %.3f ms\n", cpu_milliseconds);
    //printf("GPU Global Execution Time: %.3f ms\n", gpu_global_milliseconds);
    //printf("GPU Shared Execution Time: %.3f ms\n", gpu_shared_milliseconds);

    delete[] queryMatrix;
    delete[] keyMatrix;
    delete[] valueMatrix;
    delete[] outputCPU;
    delete[] outputGPUGlobal;
    delete[] outputGPUShared;
    delete[] attentionScoresCPU;
    delete[] attentionScoresGlobal;
    delete[] attentionScoresShared;
    delete[] transposedKeyMatrix;

    return 0;
}

