#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <thread>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#define THREAD_BLOCK_SIZE 512
#define BATCH_SIZE_POWER 27

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
 *
 * @ret: pointer to the cuda allocated output array.
 */
extern "C" int8_t* weld_ptx_execute(void *arg1, int32_t num_args, char *ptx_name, int file_name_len)
{
    /* FIXME: need to make sure arg2 is converted to appropriate form on both sides etc. */
    char trunc_ptx_name[50];
    snprintf(trunc_ptx_name, file_name_len+1, "%s", ptx_name);

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
    std::ifstream t((char *)trunc_ptx_name);
    if (!t.is_open()) {
        printf("%s not found!\n", (char *) trunc_ptx_name);
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

    /* Streaming stuff */
    int total_vals = input_args[0].num_elements;
    printf("total vals = %d\n", total_vals);
    int max_memory_elements = pow(2, 28);
    int elem_size = input_args[0].size / input_args[0].num_elements;
    printf("elem size = %d\n", elem_size);
    if (max_memory_elements > total_vals) max_memory_elements = total_vals;
    int batch_vals = pow(2, BATCH_SIZE_POWER);
    if (batch_vals > max_memory_elements) batch_vals = max_memory_elements;

    int num_chunks = ceil(total_vals / batch_vals);
    int num_streams = ceil(max_memory_elements / batch_vals);
    printf("num streams = %d\n", num_streams);
    CUstream * streams = new CUstream[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cuStreamCreate(&streams[i], CU_STREAM_DEFAULT);
    }

    for (int i = 0; i < num_args; i++) {
        // FIXME: should this be batch_vals instead?
        checkCudaErrors(cuMemAlloc(&dev_inputs[i], max_memory_elements*
                    (input_args[i].size / input_args[i].num_elements)));

        // TODO: do this later in a loop.
        //checkCudaErrors(cuMemcpyHtoD(dev_inputs[i], input_args[i].data, input_args[i].size));
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

    printf("batch vals*elem_size = %d\n", batch_vals*elem_size);
    printf("input size: %d\n", input_args[0].size);
    /* copy in the inputs for each chunk and launch kernel */
    for (int i = 0; i < num_chunks; i++) {
        int stream_index = i % num_streams;
        printf("chunk = %d\n", i);
        printf("stream index = %d\n", stream_index);
        // offset into the cpu arrays
        int offset = i * batch_vals;
        // offset into the gpu arrays
        int gpu_offset = offset % max_memory_elements;
        printf("gpu offset = %d\n", gpu_offset);
        printf("it should usually be 0 right?\n");

        // FIXME: need to copy the chunk from each of the inputs
        //checkCudaErrors(cuMemcpyHtoDAsync((CUdeviceptr ) &((double *) &dev_inputs[i])[gpu_offset], &input_args[i].data[offset], batch_vals*elem_size, streams[stream_index]));

        for (int j = 0; j < num_args; j++) {
            // TODO: correct way
            printf("copying element %d\n", j);
            checkCudaErrors(cuMemcpyHtoDAsync(dev_inputs[j], &(input_args[j].data[offset*elem_size]), batch_vals*elem_size, streams[stream_index]));
            //printf("dev_inputs[j] = %d\n", (int ) dev_inputs[j]);
            //checkCudaErrors(cuMemcpyHtoDAsync(dev_inputs[j], input_args[j].data, batch_vals*elem_size, streams[stream_index]));

            printf("element %d copied successfully\n", j);
            kernel_params[j] = (void *) &dev_inputs[j];
        }
        printf("copy successful\n");
        // TODO: need to set the correct offset in dev_output.
        kernel_params[num_args] = (void *) &dev_output;
        // TODO: in general should be this.
        // FIXME: this doesn't seem to have enough levels of indirection
        //kernel_params[num_args] = (void *) &(((double *)dev_output)[gpu_offset]);

        printf("going to launch kernel!!\n");
        struct timeval start, end, diff;
        gettimeofday(&start, NULL);
        checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                                 blockSizeX, blockSizeY, blockSizeZ,
                                 0, streams[stream_index], kernel_params, NULL));
        printf("kernel call successful!\n");
        // TODO: does it need any synchronize call here?
        gettimeofday(&end, NULL);
        timersub(&end, &start, &diff);
        printf("GPU-Kernel-Timing: %ld.%06ld\n", diff.tv_sec, diff.tv_usec);
    }

    //// Kernel launch
    //struct timeval start, end, diff;
    //gettimeofday(&start, NULL);
    //checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                             //blockSizeX, blockSizeY, blockSizeZ,
                             //0, NULL, kernel_params, NULL));
    //// TODO: does it need any synchronize call here?
    //gettimeofday(&end, NULL);
    //timersub(&end, &start, &diff);
    //printf("GPU-Kernel-Timing: %ld.%06ld\n", diff.tv_sec, diff.tv_usec);

    // Retrieve device data
    //checkCudaErrors(cuMemcpyDtoH(output, dev_output, size));

    // Clean-up
    for (int i = 0; i < num_args; i++) {
        checkCudaErrors(cuMemFree(dev_inputs[i]));
    }
    checkCudaErrors(cuModuleUnload(cudaModule));
    remove((char *) trunc_ptx_name);

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
