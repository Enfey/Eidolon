//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the evaluator, which computes what active lanes of
/// a warp compute when the warp runs a function.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "eidolon/Z3.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace mlir::eidolon {
struct LaneValue {
  z3::expr bits;
  z3::expr poison;
};

/// Represents values held by, or to be held by lanes in a warp at a given
/// point. Element k is what lane k holds. Where lane k does not run the
/// function e.g., diverges, the LaneValue is empty.
using WarpValue = SmallVector<std::optional<LaneValue>>;

/// Captures a single cause of undefined behaviour at one source location,
/// with a label and condition that holds for the exact inputs under which
/// UB occurs.
struct UndefinedBehaviour {
  std::string label;
  Location location;
  z3::expr condition;
};

struct Evaluation {
  /// Holds returned scalars, in order, representing the values of the lanes at
  /// all points of execution in the warp.
  SmallVector<WarpValue> results;
  /// Holds `UndefinedBehaviour` structs recorded by the evaluator, that
  /// correspond to source locations at which the UB occurs. Empty otherwise.
  SmallVector<UndefinedBehaviour> undefinedBehaviours;
};

/// Evaluates `function` on a warp with one lane per bit of `activeLanes`, in
/// which lthe lanes whose bits are set
/// run the function. `arguments` holds one WarpValue per argument of
/// `function`, with a value for every active lane. Fails for ops or types not
/// yet covered by the evaluator, or on a function containing more than 1 block.
FailureOr<Evaluation> evaluateFunction(FunctionOpInterface function,
                                       const BitVector &activeLanes,
                                       ArrayRef<WarpValue> arguments,
                                       z3::context &context);
} // namespace mlir::eidolon