//===----------------------------------------------------------------------===//
///
/// \file
/// This file redefines the `Z3_THROW` macro to be a thin wrapper for
/// `llvm::reportFatalInternalError`; Eidolon is compiled with -fno-exceptions
/// yet Z3's C++ API reports errors by throwing exceptions (which would be
/// ignored), motivating this workaround.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/ErrorHandling.h" // IWYU pragma: keep

#define Z3_THROW(x) llvm::reportFatalInternalError((x).msg())
#include <z3++.h> // IWYU pragma: export
