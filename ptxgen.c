#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"

namespace llvm { FunctionPass *createNVVMReflectPass(const StringMap<int>& Mapping); }

extern "C" void NVVMReflectPass(llvm::legacy::PassManager pmb);

extern "C" void NVVMReflectPass(llvm::legacy::PassManager pmb) {
  llvm::StringMap<int> reflect_mapping;
  reflect_mapping[llvm::StringRef("__CUDA_FTZ")] = 0;
  pmb.add(createNVVMReflectPass(reflect_mapping));
}

