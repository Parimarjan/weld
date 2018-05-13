#include <iostream>
#include <fstream>
#include <cassert>
#include "cuda.h"
#include <cuda_runtime.h>
#include <unistd.h>
#include <thread>
//#include <stdlib.h>
//#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define THREAD_BLOCK_SIZE 256

void checkCudaErrors(CUresult err) {
  //printf("cuda err = %d\n", (size_t)err);
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
    double *data;
    //int32_t size;
    //int32_t num_elements;
    int64_t size;
    int64_t num_elements;
} ptx_arg;

/* debug helper */
void print_vals(ptx_arg input) {
    printf("printing out vals from given input!\n");
    for (int i = 0; i < input.num_elements; i++) {
        printf("value at %d = %f;  ", i, input.data[i]);
    }
    printf("**************************************\n");
}

/*
 * TODO: update.
 * @num_elements: number of elements in the host arrays and output.
 */
extern "C" void weld_ptx_execute(void *arg1, int32_t num_args, void *arg2)
{
    /* FIXME: need to make sure arg2 is converted to appropriate form on both sides etc. */
    printf("weld ptx execute called!\n");
    ptx_arg *input_args = (ptx_arg *) arg1;
    printf("num args = %d\n", num_args);
    printf("size of 0 input: %ld\n", input_args[0].size);
    printf("num elements of 0 input: %ld\n", input_args[0].num_elements);
    /* FIXME */
    int size = input_args[0].size;
    /* FIXME: output should be of type ptx_arg too */
    int8_t *output = (int8_t *) arg2;

    // TODO: remove.
    //for (int i = 0; i < num_args; i++) print_vals(input_args[i]);
    printf("**************************************\n");
    printf("value for arg 0 = %f;\n  ", input_args[0].data[0]);
    printf("value for arg 1 = %f;\n ", input_args[1].data[0]);
    printf("**************************************\n");

    CUdevice    device;
    CUmodule    cudaModule;
    CUcontext   context;
    CUfunction  function;
    ////CUlinkState linker;       // TODO: what is the use for this?
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
    printf("Device Compute Capability: %d.%d\n", devMajor, devMinor);
    if (devMajor < 2) {
        std::cerr << "ERROR: Device 0 is not SM 2.0 or greater\n";
    }

    // TODO: this string should be passed in.
    std::ifstream t("/lfs/1/pari/kernel.ptx");
    if (!t.is_open()) {
        printf("kernel.ptx not found!\n");
        exit(0);
    }
    std::string str((std::istreambuf_iterator<char>(t)),
                std::istreambuf_iterator<char>());

    checkCudaErrors(cuCtxCreate(&context, 0, device));
    checkCudaErrors(cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0));
    checkCudaErrors(cuModuleGetFunction(&function, cudaModule, "kernel"));

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

    printf("going to set kernel params\n");
    void *kernel_params[num_args + 1];
    for (int i = 0; i < num_args; i++) {
        kernel_params[i] = (void *) &dev_inputs[i];
    }
    kernel_params[num_args] = (void *) &dev_output;

    printf("Launching kernel\n");
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
    checkCudaErrors(cuMemcpyDtoH(output, dev_output, size));

    for (int i = 0; i < num_args; i++) {
        checkCudaErrors(cuMemFree(dev_inputs[i]));
    }
    checkCudaErrors(cuMemFree(dev_output));

    // Clean-up
    checkCudaErrors(cuModuleUnload(cudaModule));
    checkCudaErrors(cuCtxDestroy(context));
}
