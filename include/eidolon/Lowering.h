//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the lowering, which yields the `LogicalResult` of running
/// `convert-gpu-to-nvvm`.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mlir::eidolon {
/// Runs `convert-gpu-to-nvvm` on each `gpu.module` nested inside `module`,
/// rewriting the module in place.
LogicalResult lowerToNVVM(ModuleOp module);
} // namespace mlir::eidolon
