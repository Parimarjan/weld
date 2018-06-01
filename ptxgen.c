#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Linker/Linker.h"


namespace llvm { FunctionPass *createNVVMReflectPass(const StringMap<int>& Mapping); }

/*extern "C" void NVVMLinkModulesWrapper(llvm::Module mod1, llvm::Module mod2);*/
extern "C" void NVVMLinkModulesWrapper(LLVMModuleRef mod1, LLVMModuleRef mod2);

extern "C" void NVVMReflectPass(llvm::legacy::PassManager pmb);

extern "C" void NVVMReflectPass(llvm::legacy::PassManager pmb) {
  llvm::StringMap<int> reflect_mapping;
  reflect_mapping[llvm::StringRef("__CUDA_FTZ")] = 0;
  pmb.add(createNVVMReflectPass(reflect_mapping));
}

/*extern "C" void NVVMLinkModulesWrapper(std::unique_ptr<llvm::Module> mod1, std::unique_ptr<llvm::Module> mod2) {*/
extern "C" void NVVMLinkModulesWrapper(LLVMModuleRef mod1, LLVMModuleRef mod2) {
  /* printf("in nvvm link modules wrapper\n"); */
  /* bool failed = llvm::Linker::linkModules(mod1, std::move(mod2), 0); */
  /* if (failed) { */
  /*    printf("linking modules failed!!\n"); */
  //}
  printf("EXTERN LIB: in nvvm link modules wrapper\n");
  llvm::Module *D = llvm::unwrap(mod2);
  std::unique_ptr<llvm::Module> M(llvm::unwrap(mod1));
  bool failed = llvm::Linker::linkModules(*D, std::move(M));
  if (failed) {
    printf("EXTERN LIB: linking modules failed!!\n");
  } else {
    printf("EXTERN LIB: SUCCESS %d", failed);
  }
}

