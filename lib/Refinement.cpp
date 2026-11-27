//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements refinement checking.
///
//===----------------------------------------------------------------------===//

#include "eidolon/Refinement.h"
#include "eidolon/Evaluate.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/BitVector.h" // IWYU pragma: keep
#include "llvm/Support/Format.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::eidolon;

/// Returns the lanes of a warp of `warpSize` lanes that run `function`.
/// For now  this is the first `width` lanes (uniform across participating
/// lanes), or the whole warp.
static FailureOr<BitVector> activeLanesOf(FunctionOpInterface function,
                                          unsigned warpSize) {
  std::optional<unsigned> width;
  for (gpu::ShuffleOp shuffle :
       function.getFunctionBody().getOps<gpu::ShuffleOp>()) {
    APInt shuffleWidth;
    if (!matchPattern(shuffle.getWidth(), m_ConstantInt(&shuffleWidth)))
      return shuffle->emitOpError("needs a constant width (temporary)");
    if (shuffleWidth.isZero() ||
        shuffleWidth.ugt(
            warpSize)) // Width 0 is not forbidden, this is temporary
      return shuffle->emitOpError() << "has a width outside 1 to " << warpSize;
    if (width && *width != shuffleWidth.getZExtValue())
      return shuffle->emitOpError("has a different width from an earlier "
                                  "shuffle, which is not yet supported");
    width = shuffleWidth.getZExtValue();
  }
  BitVector activeLanes(warpSize);
  activeLanes.set(0, width.value_or(warpSize));
  return activeLanes;
}

/// Evaluates `source` and `target` on the same lanes and inputs, and checks
/// that they return the same number of scalars. Returns a pair of the source
/// and target's `Evaluation` structs.
static FailureOr<std::pair<Evaluation, Evaluation>>
evaluateBoth(FunctionOpInterface source, FunctionOpInterface target,
             const BitVector &activeLanes, ArrayRef<WarpValue> arguments,
             z3::context &context) {
  if (target.getArgumentTypes() != source.getArgumentTypes())
    return emitError(target.getLoc(),
                     "has different argument types from its source");
  FailureOr<Evaluation> sourceEvaluation =
      evaluateFunction(source, activeLanes, arguments, context);
  if (failed(sourceEvaluation))
    return failure();
  FailureOr<Evaluation> targetEvaluation =
      evaluateFunction(target, activeLanes, arguments, context);
  if (failed(targetEvaluation))
    return failure();
  // This, generally, will incorrectly indicate the lowering does not refine the
  // source merely where the representation of the return values changes.
  if (sourceEvaluation->results.size() != source.getNumResults() ||
      targetEvaluation->results.size() != source.getNumResults())
    return emitError(target.getLoc(),
                     "returns a different number of values from its source");
  return std::make_pair(*sourceEvaluation, *targetEvaluation);
}

/// Returns a predicate that holds where `target` fails to refine `source` for a
/// given lane. The source's lane value is not poison, and the target is poison
/// or has different bits.
static z3::expr lanePred(const LaneValue &source, const LaneValue &target) {
  return !source.poison && (target.poison || target.bits != source.bits);
}

/// Prints the lanes `first` to `last` as "lane 4" or "lanes 4-12".
static void printLaneRange(raw_ostream &os, unsigned first, unsigned last) {
  if (first == last)
    os << "lane " << first;
  else
    os << "lanes " << first << "-" << last;
}

/// Prints the lanes whose bits are set in `lanes` as a range via
/// `printLaneRange()`
static void printSetLanes(raw_ostream &os, const BitVector &lanes) {
  bool firstRun = true;
  for (unsigned first = 0; first < lanes.size(); first++) {
    if (!lanes.test(first))
      continue;
    unsigned last = first;
    while (last + 1 < lanes.size() && lanes.test(last + 1))
      ++last;
    if (!firstRun)
      os << ", ";
    printLaneRange(os, first, last);
    firstRun = false;
    first = last;
  }
}

/// Prints `location` as "file:line:column" where it names a place in a file,
/// and in MLIR's own form otherwise.
static void printLocation(raw_ostream &os, Location location) {
  if (auto place = dyn_cast<FileLineColRange>(location))
    os << place.getFilename().getValue() << ":" << place.getStartLine() << ":"
       << place.getStartColumn();
  else
    os << location;
}

/// Prints `laneValue` of Type `type`. Prints poison where `laneValue` is indeed
/// poison , the number/value, or "?" where Z3 cannot reduce `laneValue` to a
/// number.
static void printLaneValue(raw_ostream &os, const LaneValue &laneValue,
                           Type type) {
  z3::expr poison = laneValue.poison.simplify();
  z3::expr bits = laneValue.bits.simplify();
  if (poison.is_true()) {
    os << "poison";
    return;
  }
  if (!poison.is_false() || !bits.is_numeral()) {
    os << "?";
    return;
  }
  unsigned bitWidth = type.getIntOrFloatBitWidth();
  APInt number(bitWidth, bits.get_numeral_uint64());
  // f16, f32, f64 (notably not higher than this due to `get_numeral_uint64()`
  // though this could be future work)
  if (auto floatType = dyn_cast<FloatType>(type))
    os << llvm::format(
        "%g", APFloat(floatType.getFloatSemantics(), number).convertToDouble());
  // i1
  else if (bitWidth == 1)
    os << (number.isOne() ? "true" : "false");
  // i16, i32, i64 (stops for same reason as above)
  else
    os << number.getSExtValue();
}

// Prints `warpValue` alongside the range of active lanes to within which
// `warpValue` is uniform e.g., "lanes 0-9: poison; lanes 9-30: 1"
static void printWarpValue(raw_ostream &os, const WarpValue &warpValue,
                           Type type) {
  SmallVector<std::string> texts;
  for (const std::optional<LaneValue> &laneValue : warpValue) {
    if (laneValue) {
      std::string text;
      llvm::raw_string_ostream stream(text);
      printLaneValue(stream, *laneValue, type);
      texts.push_back(text);
    }
  }
  bool firstRun = true;
  for (unsigned int first = 0; first < texts.size(); ++first) {
    unsigned last = first;
    // Where the index is in range, and the set of texts is uniform, increment
    // the index
    while (last + 1 < texts.size() && texts[last + 1] == texts[first])
      ++last;
    if (!firstRun)
      os << "; ";
    printLaneRange(os, first, last);
    os << ": " << texts[first];
    firstRun = false;
    first = last;
  }
}

/// Returns a `WarpValue` whose `LaneValue`s are computed from the incoming
/// `warpValue` whose `laneValue`s are Z3 formulas, possibly over the unknowns
/// (see symbolicArguments()).
static WarpValue concretise(const z3::model &example,
                            const WarpValue &warpValue) {
  WarpValue result(warpValue.size());
  for (unsigned lane = 0; lane < warpValue.size(); ++lane)
    if (warpValue[lane])
      result[lane] = LaneValue{
          // `model_completion` ensures a fixed default (0 for BV, false for
          // bool)
          // is picked if the model does not denote `poison` or `bits` so we
          // always have a concrete WarpValue to return.
          example.eval(warpValue[lane]->bits, /*model_completion=*/true),
          example.eval(warpValue[lane]->poison, /*model_completion=*/true)};
  return result;
}

/// Print a concretised `warpValue` via delegation to `concretise()` and
/// `printWarpValue` with some additional output formatting (i.e., labelling the
/// warp value given `label`).
static void printInstance(raw_ostream &os, StringRef label,
                          const WarpValue &warpValue, Type type,
                          const z3::model &example) {
  os << "     " << label << "  ";
  printWarpValue(os, concretise(example, warpValue), type);
  os << "\n";
}

/// Print the set of `arguments` (whose `WarpValue`s contain unknowns) with
/// respect to `function` via delegation to `printInstance`.
static void printArguments(raw_ostream &os, FunctionOpInterface function,
                           ArrayRef<WarpValue> arguments,
                           const z3::model &example) {
  ArrayRef<Type> types = function.getArgumentTypes();
  for (unsigned idx = 0; idx < types.size(); ++idx) {
    std::string label;
    llvm::raw_string_ostream(label)
        << "argument " << idx << " (" << types[idx] << ")";
    printInstance(os, label, arguments[idx], types[idx], example);
  }
}

/// Returns symbolic inputs for the arguments of `function`. Yields Z3
/// bitvectors for each argument for each lane, which may be set to any bits, as
/// "argI_laneK" where k is an active lane and i is the argument index with
/// respect to the function declaration.
static FailureOr<SmallVector<WarpValue>>
symbolicArguments(FunctionOpInterface function, const BitVector &activeLanes,
                  z3::context &context) {
  SmallVector<WarpValue> arguments;
  ArrayRef<Type> argumentTypes = function.getArgumentTypes();
  for (unsigned idx = 0; idx < argumentTypes.size(); ++idx) {
    if (!argumentTypes[idx].isIntOrFloat() ||
        argumentTypes[idx].getIntOrFloatBitWidth() > 64)
      return emitError(function->getLoc(),
                       "only integer and float arguments of "
                       "at most 64 bits are supported");
    WarpValue argument(activeLanes.size());
    for (unsigned lane : activeLanes.set_bits()) {
      std::string name =
          "arg" + std::to_string(idx) + "_lane" + std::to_string(lane);
      argument[lane] = LaneValue{
          context.bv_const(name.c_str(),
                           argumentTypes[idx].getIntOrFloatBitWidth()),
          context.bool_val(false)};
    }
    arguments.push_back(argument);
  }
  return arguments;
}

/// Queries `solver` on whether `condition` will hold on some input given the
/// current state of the solver. `condition` is typically a target UB condition
/// or a `lanePred` over source and target lanes. If undecided is not
/// incremented and `example` is `nullopt` then we know the condition does not
/// hold on any input left.
static std::optional<z3::model>
query(z3::solver &solver, const z3::expr &condition, unsigned &undecided) {
  solver.push();
  solver.add(condition);
  z3::check_result answer = solver.check();
  std::optional<z3::model> example;
  if (answer == z3::sat)
    example = solver.get_model();
  if (answer == z3::unknown)
    ++undecided;
  solver.pop();
  return example;
}

FailureOr<Verdict> eidolon::checkRefinement(FunctionOpInterface source,
                                            FunctionOpInterface target,
                                            const RefinementConfig &config,
                                            raw_ostream &os) {
  FailureOr<BitVector> activeLanes = activeLanesOf(source, config.warpSize);
  if (failed(activeLanes))
    return failure();
  z3::context context;
  FailureOr<SmallVector<WarpValue>> arguments =
      symbolicArguments(source, *activeLanes, context);
  if (failed(arguments))
    return failure();
  FailureOr<std::pair<Evaluation, Evaluation>> evaluations =
      evaluateBoth(source, target, *activeLanes, *arguments, context);
  if (failed(evaluations))
    return failure();
  Evaluation &sourceEvaluation = evaluations->first;
  Evaluation &targetEvaluation = evaluations->second;

  // Ideally the inverse of the UB conditions hold, as such we assume the source
  // is defined
  z3::solver solver(context);
  for (const UndefinedBehaviour &behaviour :
       sourceEvaluation.undefinedBehaviours)
    solver.add(!behaviour.condition);

  os << "  on ";
  printSetLanes(os, *activeLanes);
  os << ": ";

  // If the prior UB conditions (inversed) we assumed to be true for the source
  // cannot be satisfied then the source has undefined behaviour for all input,
  // otherwise this is `sat` and we continue
  z3::check_result anyInput = solver.check();
  if (anyInput == z3::unsat) {
    os << "the source has undefined behaviour on every input, so there is "
          "nothing to check\n";
    return Verdict::SourceAlwaysUndefined;
  }
  if (anyInput == z3::unknown) {
    os << "unknown: solver gave up\n";
    return Verdict::Unknown;
  }

  std::string findings;
  llvm::raw_string_ostream findingsStream(findings);
  unsigned undecided = 0;

  // Target UB with remaining input; if the target has UB that the source does
  // not it is checked here as the condition will be satisfied and thus we have
  // a counterexample showing the lowering induced UB for valid source inputs.
  for (const UndefinedBehaviour &behaviour :
       targetEvaluation.undefinedBehaviours) {
    if (std::optional<z3::model> example =
            query(solver, behaviour.condition, undecided)) {
      findingsStream << "  the target can have undefined behaviour: "
                     << behaviour.label << "\n    at ";
      printLocation(findingsStream, behaviour.location);
      findingsStream << "\n";
      if (source.getNumArguments() != 0) {
        findingsStream << "    example:\n";
        printArguments(findingsStream, source, *arguments, *example);
      }
    }
  }

  // Construct `lanePred` for source and target lanes as the condition to query
  // the solver with, for each of the resultant `WarpValue`s of the target and
  // source
  ArrayRef<Type> types = source.getResultTypes();
  for (unsigned idx = 0; idx < types.size(); ++idx) {
    BitVector differs(config.warpSize);
    // Example of the first lane that can differ for this result
    std::optional<z3::model> firstExample;
    for (unsigned lane : activeLanes->set_bits())
      if (std::optional<z3::model> example =
              query(solver,
                    lanePred(*sourceEvaluation.results[idx][lane],
                             *targetEvaluation.results[idx][lane]),
                    undecided)) {
        if (differs.none())
          firstExample = example;
        differs.set(lane);
      }
    if (differs.none())
      continue;
    findingsStream << "  result " << idx << " (" << types[idx]
                   << ") can differ on ";
    printSetLanes(findingsStream, differs);
    findingsStream << "\n    example for lane " << differs.find_first()
                   << " (first recorded difference)" << ":\n";
    printArguments(findingsStream, source, *arguments, *firstExample);
    printInstance(findingsStream, "source", sourceEvaluation.results[idx],
                  types[idx], *firstExample);
    printInstance(findingsStream, "target", targetEvaluation.results[idx],
                  types[idx], *firstExample);
  }

  Verdict verdict = Verdict::Refines;
  if (!findings.empty()) {
    verdict = Verdict::Counterexample;
    os << "the target does not refine the source\n";
  } else if (undecided != 0) {
    verdict = Verdict::Unknown;
    os << "unknown: solver gave up\n";
  } else {
    os << "the target refines the source\n";
  }
  os << findings;
  if (undecided != 0)
    os << "  questions the solver gave up on: " << undecided << "\n";
  return verdict;
}