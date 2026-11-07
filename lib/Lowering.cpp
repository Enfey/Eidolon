//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering by running `convert-gpu-to-nvvm`
///
//===----------------------------------------------------------------------===/

#include "../include/eidolon/Lowering.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h" // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"

using namespace mlir;

LogicalResult eidolon::lowerToNVVM(ModuleOp module) {
  // `convert-gpu-to-nvvm` is anchored on gpu.module, thus we must anchor our
  // nested pass on the same op
  PassManager passManager(module.getContext());
  passManager.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps());
  return passManager.run(module);
}
