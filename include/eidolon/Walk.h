//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the walk, which prints a function's ops in order as
/// pseudo-code.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::eidolon {
/// Prints the ops of `function` in order to `os`. Fails on any op not
/// yet known by Eidolon, and on a function without exactly one `Block`.
LogicalResult walkFunction(FunctionOpInterface function, raw_ostream &os);
} // namespace mlir::eidolon
