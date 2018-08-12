//===- llvm-link.cpp - Low-level LLVM linker ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-link a.bc b.bc c.bc -o x.bc
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker/Linker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/FunctionInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Object/FunctionIndexObjectFile.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/IPO.h"

#include <memory>
#include <iostream>
#include <iostream>
#include <fstream>

namespace llvm { extern FunctionPass *createNVVMReflectPass(const StringMap<int>& Mapping);}

using namespace llvm;


extern "C" int link(int argc, const char **argv, const char *kernel_buffer);

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
	       cl::desc("<input bitcode files>"));

static cl::list<std::string> OverridingInputs(
					      "override", cl::ZeroOrMore, cl::value_desc("filename"),
					      cl::desc(
						       "input bitcode file which can override previously defined symbol(s)"));

// Option to simulate function importing for testing. This enables using
// llvm-link to simulate ThinLTO backend processes.
static cl::list<std::string> Imports(
				     "import", cl::ZeroOrMore, cl::value_desc("function:filename"),
				     cl::desc("Pair of function name and filename, where function should be "
					      "imported from bitcode in filename"));

// Option to support testing of function importing. The function index
// must be specified in the case were we request imports via the -import
// option, as well as when compiling any module with functions that may be
// exported (imported by a different llvm-link -import invocation), to ensure
// consistent promotion and renaming of locals.
static cl::opt<std::string> FunctionIndex("functionindex",
					  cl::desc("Function index filename"),
					  cl::init(""),
					  cl::value_desc("filename"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"), cl::init("-"),
	       cl::value_desc("filename"));

static cl::opt<bool>
Internalize("internalize", cl::desc("Internalize linked symbols"));

static cl::opt<bool>
OnlyNeeded("only-needed", cl::desc("Link only needed symbols"));

static cl::opt<bool>
Force("f", cl::desc("Enable binary output on terminals"));

static cl::opt<bool>
OutputAssembly("S",
	       cl::desc("Write output as LLVM assembly"), cl::Hidden);

static cl::opt<bool>
Verbose("v", cl::desc("Print information about actions taken"));

static cl::opt<bool>
SuppressWarnings("suppress-warnings", cl::desc("Suppress all linking warnings"),
		 cl::init(false));

static cl::opt<bool>
PreserveModules("preserve-modules",
		cl::desc("Preserve linked modules for testing"));

static cl::opt<bool> PreserveBitcodeUseListOrder(
						 "preserve-bc-uselistorder",
						 cl::desc("Preserve use-list order when writing LLVM bitcode."),
						 cl::init(true), cl::Hidden);

static cl::opt<bool> PreserveAssemblyUseListOrder(
						  "preserve-ll-uselistorder",
						  cl::desc("Preserve use-list order when writing LLVM assembly."),
						  cl::init(false), cl::Hidden);

// Read the specified bitcode file in and return it. This routine searches the
// link path for the specified file to try to find it...
static std::unique_ptr<Module> loadFile(const std::string &FN,
					LLVMContext &Context,
					bool MaterializeMetadata = true) {
  SMDiagnostic Err;
  if (Verbose) errs() << "Loading '" << FN << "'\n";
  std::unique_ptr<Module> Result =
    getLazyIRFileModule(FN, Err, Context, !MaterializeMetadata);
  if (!Result)
    Err.print("linker", errs());

  if (MaterializeMetadata) {
    Result->materializeMetadata();
    UpgradeDebugInfo(*Result);
  }

  return Result;
}

// Read the specified assembly string and return it.
static std::unique_ptr<Module> loadBuffer(const char* kernel_str,
					  LLVMContext &Context,
					  bool MaterializeMetadata = true) {
  SMDiagnostic Err;
  std::string content(kernel_str);
  llvm::StringRef str_ref = llvm::StringRef(content);
  std::unique_ptr<Module> Result = llvm::parseAssemblyString(str_ref, Err, Context);
  if (!Result)
    std::cout << "An error in parsing Assembly string";

  if (MaterializeMetadata) {
    Result->materializeMetadata();
    UpgradeDebugInfo(*Result);
  }

  return Result;
}

static void diagnosticHandler(const DiagnosticInfo &DI) {
  unsigned Severity = DI.getSeverity();
  switch (Severity) {
  case DS_Error:
    errs() << "ERROR: ";
    break;
  case DS_Warning:
    if (SuppressWarnings)
      return;
    errs() << "WARNING: ";
    break;
  case DS_Remark:
  case DS_Note:
    llvm_unreachable("Only expecting warnings and errors");
  }

  DiagnosticPrinterRawOStream DP(errs());
  DI.print(DP);
  errs() << '\n';
}

static void diagnosticHandlerWithContext(const DiagnosticInfo &DI, void *C) {
  diagnosticHandler(DI);
}
static bool linkModule(std::unique_ptr<Module> M,
		       Linker &L,
		       unsigned ApplicableFlags){
  if (!M.get()) {
      errs() << "linker : error loading file \n";
      return false;
  }
  
  if (verifyModule(*M, &errs())) {
    errs() << " linker : error: input module is broken!\n";
    return false;
  }

  // If a function index is supplied, load it so linkInModule can treat
  // local functions/variables as exported and promote if necessary.
  std::unique_ptr<FunctionInfoIndex> Index;
  if (!FunctionIndex.empty()) {
    ErrorOr<std::unique_ptr<FunctionInfoIndex>> IndexOrErr =
      llvm::getFunctionIndexForFile(FunctionIndex, diagnosticHandler);
    std::error_code EC = IndexOrErr.getError();
    if (EC) {
      errs() << EC.message() << '\n';
      return false;
    }
    Index = std::move(IndexOrErr.get());
  }
  
  if (L.linkInModule(std::move(M), ApplicableFlags, Index.get()))
    return false;
  
  // If requested for testing, preserve modules by releasing them from
  // the unique_ptr before the are freed. This can help catch any
  // cross-module references from e.g. unneeded metadata references
  // that aren't properly set to null but instead mapped to the source
  // module version. The bitcode writer will assert if it finds any such
  // cross-module references.
  if (PreserveModules)
    M.release();

  return true;
}

static bool linkFiles(LLVMContext &Context,
		      Linker &L,
		      const cl::list<std::string> &Files,
		      unsigned Flags,
		      const char* kernel_str) {
  // Filter out flags that don't apply to the first file we load.
  unsigned ApplicableFlags = Flags & Linker::Flags::OverrideFromSrc;

  // Load the llvm kernel first as a module
  std::unique_ptr<Module> M = loadBuffer(kernel_str, Context);
  linkModule(std::move(M), L, ApplicableFlags);

  for (const auto &File : Files) {
    std::unique_ptr<Module> M = loadFile(File, Context);  
    linkModule(std::move(M), L, ApplicableFlags);
  }
  return true;
}

int link(int argc, const char **argv, const char* kernel_str) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  // Initialize LLVMContext
  LLVMContext &Context = getGlobalContext();
  Context.setDiagnosticHandler(diagnosticHandlerWithContext, nullptr, true);

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // the following cannot be commented out
  // it modifies a static GlobalParser which all other following
  // functions use by indirectly invoking cl::xyz functions
  cl::ParseCommandLineOptions(argc, argv, "llvm linker\n");


  llvm::Module Composite("llvm-link", Context);
  Linker L(Composite);

  unsigned Flags = Linker::Flags::None;
  if (Internalize)
    printf("Internalized");
    Flags |= Linker::Flags::InternalizeLinkedSymbols;
  if (OnlyNeeded)
    Flags |= Linker::Flags::LinkOnlyNeeded;

  // First add all the regular input files
  if (!linkFiles(Context, L, InputFilenames, Flags, kernel_str))
    return 1;

  std::error_code EC;
  tool_output_file Out(OutputFilename, EC, sys::fs::F_None);
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }


  if (verifyModule(Composite, &errs())) {
    errs() << argv[0] << ": error: linked module is broken!\n";
    return 1;
  }


  // Run internalize pass
  llvm::legacy::PassManager PM;
  const char* ExportList = "kernel";
  ModulePass* modpass = llvm::createInternalizePass(ExportList);
  PM.add(modpass);
  
  // NVVM Reflect Pass
  llvm::StringMap<int> reflect_mapping;
  reflect_mapping[llvm::StringRef("__CUDA_FTZ")] = 1;
  PM.add(createNVVMReflectPass(reflect_mapping));

  // O3 pass TODO
  
  PM.run(Composite);

  

  
  if (Verbose) errs() << "Writing bitcode...\n";

  if (OutputAssembly) {
    Composite.print(Out.os(), nullptr, PreserveAssemblyUseListOrder);
  } else if (Force || !CheckBitcodeOutputToConsole(Out.os(), true)) {
    WriteBitcodeToFile(&Composite, Out.os(), PreserveBitcodeUseListOrder);
  } else {
  }
  // Declare success.
  Out.keep();

  return 0;
}

