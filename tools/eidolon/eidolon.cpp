//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the eidolon tool, which parses a module, lowers a copy
/// of the module, then walks the original's functions alongside the lowered's
/// functions.
///
//===----------------------------------------------------------------------===//

#include "eidolon/Lowering.h"
#include "eidolon/Refinement.h"
#include "eidolon/Walk.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string>
  inputFilename(cl::Positional, // Would expect a flag otherwise
                cl::desc("<input file>"), cl::Required);

static cl::opt<bool> printLowered("print-lowered",
                                  cl::desc("Prints the lowered module instead of walking it"));

static cl::opt<bool>
    printWalk("print-walk", cl::desc("Prints source and target function pairs "
                                     "as pseudocode and does not validate"));

static cl::opt<unsigned>
    solverTimeout("solver-timeout",
                  cl::desc("Limits the time the solver spends to solve a "
                           "particular condition (infinite by default)."),
                  cl::value_desc("milliseconds"), cl::init(0));

/// Prints `source` and `target` labels and then checks that `target` refines
/// source according to `eidlon::checkRefinement`. If --print-walk was passed
/// then the pseudocode is sent to `outs()`. Returns whether the pair 'passes'
/// i.e., the `target` refines the `source` and whether the walk failed or
/// succeeded should the flag be passed.
static FailureOr<bool> processPair(FunctionOpInterface source,
                                   FunctionOpInterface target,
                                   const eidolon::RefinementConfig &config) {
  outs() << "@" << source.getName() << "\n";
  if (printWalk) {
    outs() << "  source\n";
    if (failed(eidolon::walkFunction(source, outs())))
      return failure();
    outs() << "  target\n";
    if (failed(eidolon::walkFunction(target, outs())))
      return failure();
    return true;
  }
  FailureOr<eidolon::Verdict> verdict =
      eidolon::checkRefinement(source, target, config, outs());
  if (failed(verdict))
    return failure();
  return *verdict == eidolon::Verdict::Refines;
}

static LogicalResult eidolonExec(MLIRContext &context) {
  SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler diagnosticHandler(sourceMgr, &context);

  OwningOpRef<ModuleOp> source =
      parseSourceFile<ModuleOp>(inputFilename, sourceMgr, &context);
  if (!source)
    return failure();

  // The lowering pass will rewrite the module, thus we must instantiate a clone
  // to have the module to compare against
  OwningOpRef<ModuleOp> target = source->clone();
  if (failed(eidolon::lowerToNVVM(*target)))
    return failure();

  if (printLowered) {
    // -> is overloaded to yield the underlying op (and the print method)
    target->print(outs());
    outs() << "\n";
    return success();
  }

  // Pair each func.func in the source with the llvm.func it was lowered to and
  // walk both
  eidolon::RefinementConfig config;
  config.solverTimeoutMs = solverTimeout;
  unsigned pairsWalked = 0;
  bool allRefine = true;
  for (gpu::GPUModuleOp sourceGpuModule : source->getOps<gpu::GPUModuleOp>()) {
    for (func::FuncOp sourceFunction : sourceGpuModule.getOps<func::FuncOp>()) {
      // Construct hierarchical symbol name for the `FuncOp`, e.g.,
      // @test_module_4::@gpu_shuffle
      auto fullName =
          SymbolRefAttr::get(sourceGpuModule.getSymNameAttr(),
                             {FlatSymbolRefAttr::get(sourceFunction)});
      auto targetFunction = target->lookupSymbol<LLVM::LLVMFuncOp>(
          fullName); // `sym_name` is retained
      if (!targetFunction) {
        return emitError(sourceFunction.getLoc())
               << "no llvm.func named " << fullName << " after lowering";
      }
      ++pairsWalked;
      FailureOr<bool> passes =
          processPair(sourceFunction, targetFunction, config);
      if (failed(passes))
        return failure();
      if (!*passes)
        allRefine = false;
    }
  }

  // An input with nothing to compare must not pass silently
  if (pairsWalked == 0)
    return emitError(source->getLoc(),
                     "nothing to check: no func.func inside a gpu.module");
  return success(allRefine);
}

int main(int argc, char **argv) {
  InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Eidolon\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  MLIRContext context(registry);
  return failed(eidolonExec(context));
}
