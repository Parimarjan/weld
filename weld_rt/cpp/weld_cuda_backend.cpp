#include <iostream>
#include <fstream>
#include <cassert>
//#include "cuda.h"
//#include "cuda/cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <thread>
//#include <stdlib.h>
//#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define THREAD_BLOCK_SIZE 512

void checkCudaErrors(CUresult err) {
  if (err != CUDA_SUCCESS) {
    //printf("cuda success failure!!\n");
    //char errMsg[10000];
    const char *errMsg = (char *) malloc(10000);
    const char **errMsgptr = &errMsg;
    cuGetErrorString (err, errMsgptr);
    printf("cuda string error: %s\n", *errMsgptr);
  } else {
    //printf("cuda init actually worked!\n");
  }
  assert(err == CUDA_SUCCESS);
}

typedef struct ptx_arg {
    // FIXME: type of this should not affect anything.
    uint8_t *data;
    //int32_t size;
    //int32_t num_elements;
    int64_t size;
    int64_t num_elements;
} ptx_arg;

/* debug helper */
void print_vals(ptx_arg input) {
    printf("printing out vals from given input!\n");
    for (int i = 0; i < input.num_elements; i++) {
        printf("value at %d = %f;  ", i, (double ) input.data[i]);
    }
    printf("**************************************\n");
}

/*
 * TODO: update.
 * @arg1:
 * @num_args: number of elements in the host arrays and output.
 * @arg2: pointer to the output array since this does not need to be a ptx_arg struct.
 *
 * @ret: pointer to the cuda allocated output array.
 */
//extern "C" int8_t* weld_ptx_execute(void *arg1, int32_t num_args, void *arg2)
extern "C" int8_t* weld_ptx_execute(void *arg1, int32_t num_args)
{
    /* FIXME: need to make sure arg2 is converted to appropriate form on both sides etc. */
    ptx_arg *input_args = (ptx_arg *) arg1;
    /* FIXME */
    int size = input_args[0].size;
    /* FIXME: output should be of type ptx_arg too */
    //int8_t *output = (int8_t *) arg2;

    CUdevice    device;
    CUmodule    cudaModule;
    CUcontext   context;
    CUfunction  function;
    int         devCount;
    // CUDA initialization
    // TODO: maybe this does not have to be reinitialized every time?
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGetCount(&devCount));
    checkCudaErrors(cuDeviceGet(&device, 0));

    char name[128];
    checkCudaErrors(cuDeviceGetName(name, 128, device));
    printf("Using CUDA device %s\n", name);

    int devMajor, devMinor;
    checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, device));
    //printf("Device Compute Capability: %d.%d\n", devMajor, devMinor);
    if (devMajor < 2) {
        std::cerr << "ERROR: Device 0 is not SM 2.0 or greater\n";
    }

    // TODO: this string should be passed in.
    std::ifstream t("/tmp/kernel.ptx");
    if (!t.is_open()) {
        printf("kernel.ptx not found!\n");
        exit(0);
    }
    std::string str((std::istreambuf_iterator<char>(t)),
                std::istreambuf_iterator<char>());

    struct timeval start_compile, end_compile, end_ctx, end_mod, diff_compile;
    gettimeofday(&start_compile, NULL);
    checkCudaErrors(cuCtxCreate(&context, 0, device));
    gettimeofday(&end_ctx, NULL);
    timersub(&end_ctx, &start_compile, &diff_compile);
    printf("CUDA-ctx-create-Timing: %ld.%06ld\n", diff_compile.tv_sec, diff_compile.tv_usec);

    checkCudaErrors(cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0));
    gettimeofday(&end_compile, NULL);
    checkCudaErrors(cuModuleGetFunction(&function, cudaModule, "kernel"));

    timersub(&end_compile, &start_compile, &diff_compile);
    printf("CUDA-Compile-Timing: %ld.%06ld\n", diff_compile.tv_sec, diff_compile.tv_usec);

    CUdeviceptr dev_output;
    /* FIXME: this should not be based on input args */
    checkCudaErrors(cuMemAlloc(&dev_output, input_args[0].size));
    CUdeviceptr dev_inputs[num_args];
    for (int i = 0; i < num_args; i++) {
        checkCudaErrors(cuMemAlloc(&dev_inputs[i], input_args[i].size));
        checkCudaErrors(cuMemcpyHtoD(dev_inputs[i], input_args[i].data, input_args[i].size));
    }

    /* FIXME: be more flexible about dimensions? */
    unsigned blockSizeX = THREAD_BLOCK_SIZE;
    unsigned blockSizeY = 1;
    unsigned blockSizeZ = 1;

    // TODO: we are implicitly assuming all of same num elements?
    unsigned gridSizeX  = (size_t) ceil((float) input_args[0].num_elements / (float) THREAD_BLOCK_SIZE);
    unsigned gridSizeY  = 1;
    unsigned gridSizeZ  = 1;

    void *kernel_params[num_args + 1];
    for (int i = 0; i < num_args; i++) {
        kernel_params[i] = (void *) &dev_inputs[i];
    }
    kernel_params[num_args] = (void *) &dev_output;

    //// Kernel launch
    struct timeval start, end, diff;
    gettimeofday(&start, NULL);
    checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                             blockSizeX, blockSizeY, blockSizeZ,
                             0, NULL, kernel_params, NULL));
    // TODO: does it need any synchronize call here?
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    printf("GPU-Kernel-Timing: %ld.%06ld\n", diff.tv_sec, diff.tv_usec);

    // Retrieve device data
    //checkCudaErrors(cuMemcpyDtoH(output, dev_output, size));

    // Clean-up
    for (int i = 0; i < num_args; i++) {
        checkCudaErrors(cuMemFree(dev_inputs[i]));
    }
    checkCudaErrors(cuModuleUnload(cudaModule));
    remove("/tmp/kernel.ptx");

    // Note: we copy dev_output back to host memory later (so we can potentially act on it in gpu
    // memory, e.g., by passing it to thrust). Thus we don't free these here.
    // if we need to copy it back, seems like we need to leave this around?
    //checkCudaErrors(cuCtxDestroy(context));
    //checkCudaErrors(cuMemFree(dev_output));

    return (int8_t *) dev_output;
}

// FIXME: maybe we don't need a separate function for this.
extern "C" void weld_copy_dtoh(int8_t *host, int8_t *dev, int size) {
    // Retrieve device data
    checkCudaErrors(cuMemcpyDtoH(host, (CUdeviceptr) dev, size));
    checkCudaErrors(cuMemFree((CUdeviceptr) dev));
};
