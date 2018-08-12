#include <iostream>

extern "C" int link(int argc, const char **argv, const char *kernel_buffer);

extern "C" void link_libdevice(const char* kernel_str);

// Links the llvm module passed in as a string with the nvvm libdevice bitcode module.
extern "C" void link_libdevice(const char* kernel_str){
  std::string CUDA_PATH_str = getenv("CUDA_PATH");
  std::string cuda_path_arg = CUDA_PATH_str +  "/nvvm/libdevice/libdevice.compute_50.10.bc";

  if (kernel_str == NULL){
    printf("kernel_str is NULL\n");
  }

  const char **argsv = new const char*[5];
  argsv[0] = "./ptxgen";
  argsv[1] = cuda_path_arg.c_str();
  argsv[2] = "-o";
  argsv[3] = "t2.linked.bc";
  argsv[4] = NULL;

  link(4, argsv, kernel_str);
}
