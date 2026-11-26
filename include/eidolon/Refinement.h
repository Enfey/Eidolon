//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the comparison of a source function with the function in
/// the dialect it was lowered to.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::eidolon {
/// The configuration for one refinement check
struct RefinementConfig {
  /// The number of lanes in a warp
  unsigned warpSize = 32;
};

/// The outcome of a single refinement check
enum class Verdict {
  /// Yielded where across all inputs where the source is defined, the target
  /// is not undefined, and each of the results are equal across all lanes,
  /// except in the case of poison.
  Refines,
  /// Yielded where some input where the source is defined can make the target
  /// undefined, or give a result that differs from the source's on some lane.
  Counterexample,
  /// Yielded where the solver gave up on at least one question, and no
  /// counterexample
  /// was produced.
  Unknown,
  /// The source has undefined behaviour on all inputs, thus there is nothing to
  /// check.
  SourceAlwaysUndefined
};

/// Searches for inputs under which `target` fails to refine `source` over a
/// configuration `config`, with every argument of every active lane declared as
/// Z3 unknowns. Prints either that `target` refines `source` for all inputs, or
/// that a counterexample was produced. Fails where the evaluator cannot support
/// one of either the `source` or `target`.
FailureOr<Verdict> checkRefinement(FunctionOpInterface source,
                                   FunctionOpInterface target,
                                   const RefinementConfig &config,
                                   raw_ostream &os);
} // namespace mlir::eidolon
