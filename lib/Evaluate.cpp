//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the evaluator.
///
//===----------------------------------------------------------------------===//

#include "eidolon/Evaluate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h" // IWYU pragma: keep
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::eidolon;

/// Yields the type of each scalar value of type `type`. Works for struct
/// types(destructured). Fails on non-integers and non-floats. We will need to
/// generalise to unpack more types in future
static FailureOr<SmallVector<Type>> scalarTypesOf(Type type) {
  if (type.isIntOrFloat())
    return SmallVector<Type>{type};
  auto structType = dyn_cast<LLVM::LLVMStructType>(type);
  if (!structType)
    return failure();
  ArrayRef<Type> fields =
      structType.getBody(); // Context maintains precisely one copy of each
                            // distinct type, so we get refs back to them
  if (!all_of(fields, [](Type field) { return field.isIntOrFloat(); }))
    return failure();
  return SmallVector<Type>(fields);
}

namespace {
/// Evaluates one function on the active lanes of a warp. SSA values are denoted
/// as one WarpValue. An evaluator lives for one function.
class FunctionEvaluator {
public:
  FunctionEvaluator(const BitVector &activeLanes, z3::context &context)
      : activeLanes(activeLanes), context(context),
        undefinedBehaviour(context.bool_val(false)) {};

  FailureOr<Evaluation> evaluate(FunctionOpInterface function,
                                 ArrayRef<WarpValue> arguments);

private:
  /// Holds scalars of an SSA value across participating lanes at a given point
  /// of the function.
  using SSAScalars = SmallVector<WarpValue>;

  /// The semantics of an operation that executes independently on each lane.
  /// Given the bits of the operation's operands on one lane, return the bits of
  /// the result on that lane. This does not represent the semantics of
  /// cross-lane operations e.g., warp shuffling. The result may be poison.
  using LaneSemantics = function_ref<LaneValue(ArrayRef<z3::expr>)>;

  /// Returns the number of lanes in the warp
  unsigned warpSize() const { return activeLanes.size(); }

  /// Returns the lane ID of lane `lane` as a Z3 bit-vector (interoperability
  /// reasons motivate expansion despite the fact we can represent IDs with 7
  /// bits)
  z3::expr laneID(unsigned lane) const { return context.bv_val(lane, 32); }

  /// Returns how many bits `value` has
  unsigned bitWidthOf(const WarpValue &value) const;

  /// Returns a `WarpValue` constructed from `value` for each active lane
  WarpValue uniform(const LaneValue &value) const;

  /// Records the lanewise scalars of a value the evaluation has just computed
  void define(Value value, SSAScalars scalars);

  /// Returns the scalars of a value - use where `value` is a struct
  SSAScalars scalarsOf(Value value) const;

  /// Returns the singular scalar of an integer or float SSA value - implemented
  /// as a thin-wrapper of FunctionEvaluator::scalarsOf
  WarpValue scalarOf(Value value) const;

  /// Returns a new Z3 constant of `bitWidth` bits, representing an undefined
  /// result
  z3::expr undefined(unsigned bitWidth);

  /// Returns what lane `lane` holds in `value`, where `lane` may be any 32-bit
  /// term. Reading a lane that is not active yields an undefined value.
  LaneValue readLane(const WarpValue &value, const z3::expr &lane);

  void applyLaneOp(Operation *op, LaneSemantics semantics);

  LogicalResult defineConstant(Operation *op, Attribute constant);

  LogicalResult evaluateOp(Operation *op);

  const BitVector &activeLanes;
  z3::context &context;
  DenseMap<Value, SSAScalars> values;
  SmallVector<WarpValue> results;
  z3::expr undefinedBehaviour;
  unsigned undefinedCount = 0;
};
} // namespace

unsigned FunctionEvaluator::bitWidthOf(const WarpValue &value) const {
  // Every active lane holds will hold `value`, thus first active lane is fine
  return value[activeLanes.find_first()]->bits.get_sort().bv_size();
}

WarpValue FunctionEvaluator::uniform(const LaneValue &value) const {
  WarpValue result(warpSize());
  for (unsigned lane : activeLanes.set_bits())
    result[lane] = value;
  return result;
}

void FunctionEvaluator::define(Value value, SSAScalars scalars) {
  values[value] = std::move(scalars);
}

FunctionEvaluator::SSAScalars FunctionEvaluator::scalarsOf(Value value) const {
  assert(values.contains(value) && "value not defined by evaluator");
  return values.lookup(value);
}

WarpValue FunctionEvaluator::scalarOf(Value value) const {
  SSAScalars scalar = scalarsOf(value);
  assert(scalar.size() == 1 && "expected an integer or float");
  return scalar.front();
}

z3::expr FunctionEvaluator::undefined(unsigned bitWidth) {
  std::string name = "undefined" + std::to_string(undefinedCount++);
  return context.bv_const(name.c_str(), bitWidth); // Unknown uniquely named bv
}

LaneValue FunctionEvaluator::readLane(const WarpValue &value,
                                      const z3::expr &lane) {
  LaneValue result{undefined(bitWidthOf(value)),
                   context.bool_val(false)}; // Default
  for (unsigned candidate : activeLanes.set_bits()) {
    z3::expr isSourceLane =
        lane ==
        laneID(candidate); // We cannot use `lane` as an index directly because
                                // we do not know exactly, what `lane` is.
    unsigned sourceLane = candidate;
    result = {z3::ite(isSourceLane, value[sourceLane]->bits, result.bits),
              z3::ite(isSourceLane, value[sourceLane]->poison, result.poison)};
  }
  return result;
}

void FunctionEvaluator::applyLaneOp(Operation *op, LaneSemantics semantics) {
  SmallVector<WarpValue> operands;
  for (Value operand : op->getOperands())
    operands.push_back(scalarOf(operand));
  WarpValue result(warpSize());
  for (unsigned lane : activeLanes.set_bits()) {
    SmallVector<z3::expr> bits;
    for (const WarpValue &operand : operands)
      bits.push_back(operand[lane]->bits);  // Take the bits for the operand for
                                                // the specific active lane
    LaneValue laneResult = semantics(bits); // Apply the semantics to this lanes
                                            // operands, yielding the result
    // The result is poison on any lane where an operand is poison
    for (const WarpValue &operand : operands)
      laneResult.poison = laneResult.poison || operand[lane]->poison;
    result[lane] = laneResult;
  }
  define(op->getResult(0), {result}); // Unsure if 0 will generalise
}

LogicalResult FunctionEvaluator::defineConstant(Operation *op,
                                                Attribute constant) {
  APInt bits;
  if (auto integer = dyn_cast<IntegerAttr>(
          constant)) // TODO: Why you can do this with -fno-rtti (generally I
                     // understand why, just want to become a little more
                     // confident)
    bits = integer.getValue();
  else if (auto floating = dyn_cast<FloatAttr>(constant))
    bits = floating.getValue().bitcastToAPInt();
  else
    return op->emitOpError("kind of attribute is not yet supported");
  // .getZExtValue() would trigger an assertion otherwise, we'd like to
  // centralise the error handling
  if (bits.getBitWidth() > 64)
    return op->emitOpError("attribute wider than 64 bits is not yet supported");
  LaneValue laneValue{context.bv_val(bits.getZExtValue(), bits.getBitWidth()),
                      context.bool_val(false)};
  define(op->getResult(0), {uniform(laneValue)});
  return success();
}

LogicalResult FunctionEvaluator::evaluateOp(Operation *op) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      // Source ops: TODO // gpu::BallotOp, gpu::SubgroupReduceOp,
      // gpu::SubgroupBroadcastOp...
      .Case<arith::ConstantOp>([&](arith::ConstantOp constant) {
        return defineConstant(constant, constant.getValue());
      })
      .Case<gpu::ShuffleOp>([&](gpu::ShuffleOp shuffle) {
        WarpValue shuffled = scalarOf(shuffle.getValue());
        WarpValue offset = scalarOf(shuffle.getOffset());
        WarpValue width = scalarOf(shuffle.getWidth());
        WarpValue result(warpSize()), valid(warpSize());
        for (unsigned lane : activeLanes.set_bits()) {
          z3::expr laneOffset = offset[lane]->bits;
          z3::expr k = laneID(lane);
          z3::expr from = k;
          switch (shuffle.getMode()) {
          case gpu::ShuffleMode::XOR:
            from = k ^ laneOffset;
            break;
          case gpu::ShuffleMode::UP:
            from = k - laneOffset;
            break;
          case gpu::ShuffleMode::DOWN:
            from = k + laneOffset;
            break;
          case gpu::ShuffleMode::IDX:
            from = laneOffset;
            break;
          }
          // The result is valid where the lane it comes from is one of the
          // first `width` lanes; otherwise the result is poison.
          z3::expr inRange =
              z3::sge(from, 0) && z3::slt(from, width[lane]->bits);
          z3::expr operandPoison = offset[lane]->poison || width[lane]->poison;
          LaneValue read = readLane(shuffled, from);
          result[lane] =
              LaneValue{read.bits, !inRange || read.poison || operandPoison};
          valid[lane] = LaneValue{
              z3::ite(inRange, context.bv_val(1, 1), context.bv_val(0, 1)),
              operandPoison};
          // o > w is defined as UB - we do not know what that lane will contain
          undefinedBehaviour =
              undefinedBehaviour || z3::uge(laneOffset, warpSize());
        }
        define(shuffle.getShuffleResult(), {result});
        define(shuffle.getValid(), {valid});
        return success();
      })
      // Target ops.
      .Case<LLVM::ConstantOp>([&](LLVM::ConstantOp constant) {
        return defineConstant(constant, constant.getValue());
      })
      .Case<LLVM::SubOp>([&](LLVM::SubOp sub) -> LogicalResult {
        if (sub.getOverflowFlags() != LLVM::IntegerOverflowFlags::none)
          return sub->emitOpError("with overflow flags is not yet supported");
        applyLaneOp(sub, [&](ArrayRef<z3::expr> bits) {
          return LaneValue{bits[0] - bits[1], context.bool_val(false)};
        });
        return success();
      })
      .Case<LLVM::LShrOp>([&](LLVM::LShrOp shift) -> LogicalResult {
        if (shift.getIsExact())
          return shift->emitOpError("with the exact flag is not yet supported");
        unsigned bitWidth = shift.getType().getIntOrFloatBitWidth();
        applyLaneOp(shift, [&](ArrayRef<z3::expr> bits) {
          // Shifting by operand >= `bitWidth` yields a mask that is poison
          return LaneValue{z3::lshr(bits[0], bits[1]),
                           z3::uge(bits[1], bitWidth)};
        });
        return success();
      })
      .Case<NVVM::ShflOp>([&](NVVM::ShflOp shuffle) -> LogicalResult {
        if (warpSize() != 32)
          return shuffle->emitOpError("only a warp size of 32 is valid");
        WarpValue mask = scalarOf(shuffle.getThreadMask());
        WarpValue shuffled = scalarOf(shuffle.getVal());
        WarpValue offset = scalarOf(shuffle.getOffset());
        WarpValue clamp = scalarOf(shuffle.getMaskAndClamp());
        unsigned bitWidth = bitWidthOf(shuffled);
        // Lanes that do not run the shuffle, 1 bit per lane at bit `i` for lane
        // `k`
        uint32_t inactiveBits = 0;
        for (unsigned lane = 0; lane < 32; lane++)
          if (!activeLanes.test(lane))
            inactiveBits |= uint32_t(1) << lane;
        WarpValue result(warpSize()), valid(warpSize());
        for (unsigned lane : activeLanes.set_bits()) {
          // This follows the shfl.sync semantics defined in the PTX manual,
          // where `inRange` is called pval.
          /// Source :
          /// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync
          z3::expr k = laneID(lane);
          z3::expr b = offset[lane]->bits & 31; // b[4:0]
          z3::expr c =
              clamp[lane]->bits & 31; // c[4:0] - singular boundary at some lane
          // c[12:8] - defines a sub-group of the warp (not to be confused with
          // `ThreadMask`) MLIR passes the mask portion of `MaskAndClamp` as
          // zero, but other sources may still set this Example:
          //                        [12:8][7:5][4:0]
          //  clamp[lane]->bits = ..11000  000  00111 >> 8 & 31 = 11000
          //  the number of sub-groups in the warp is defined by 2^b, where b is
          //  the number of bits set (this is CUDA/PTX specific)
          z3::expr segmentMask = z3::lshr(clamp[lane]->bits, 8) & 31;
          z3::expr maxLane = (k & segmentMask) | (c & ~segmentMask);
          z3::expr minLane =
              k & segmentMask; // e.g., k = 00011, 00011 & 11000 = 0 , yields
                               // seg num you start in
          z3::expr from = k;
          z3::expr inRange = context.bool_val(false);
          switch (shuffle.getKind()) {
          case NVVM::ShflKind::up:
            from = k - b;
            inRange = z3::sge(from, maxLane);
            break;
          case NVVM::ShflKind::down:
            from = k + b;
            inRange = z3::sle(from, maxLane);
            break;
          case NVVM::ShflKind::bfly:
            from = k ^ b;
            inRange = z3::sle(from, maxLane);
            break;
          case NVVM::ShflKind::idx:
            from = minLane |
                   (b & ~segmentMask); // Discard bits of `b` where
                                       // `segmentMask` is set such that `b`
                                       // cannot access a different sub-group.
            inRange = z3::sle(from, maxLane);
            break;
          }
          from = z3::ite(inRange, from, k);

          // A lane outside the mask that runs the shuffle, or a lane in the
          // mask that never runs it, is UB. Similarly, reading a lane outside
          // the mask gives an undefined result.
          z3::expr laneMask = mask[lane]->bits;
          undefinedBehaviour =
              undefinedBehaviour || laneMask.extract(lane, lane) ==
                                        0; // Is executing, where the mask says
                                           // the lane should not be taking part
          if (inactiveBits != 0)
            undefinedBehaviour =
                undefinedBehaviour ||
                (laneMask & context.bv_val(inactiveBits, 32)) != 0;
          z3::expr inMask = z3::lshr(laneMask, from).extract(0, 0) == 1; // Valid read, lane is participating
          LaneValue read = readLane(shuffled, from);
          z3::expr operandPoison =
              mask[lane]->poison || offset[lane]->poison || clamp[lane]->poison;
          result[lane] =
              LaneValue{z3::ite(inMask, read.bits, undefined(bitWidth)),
                        (inMask && read.poison) || operandPoison};
          valid[lane] = LaneValue{
              z3::ite(inRange, context.bv_val(1, 1), context.bv_val(0, 1)),
              operandPoison};
        }
        if (shuffle.getReturnValueAndIsValidAttr())
          define(shuffle.getResult(), {result, valid});
        else
          define(shuffle.getResult(), {result});
        return success();
      })
      .Case<LLVM::ExtractValueOp>(
          [&](LLVM::ExtractValueOp extract) -> LogicalResult {
            ArrayRef<int64_t> position = extract.getPosition();
            if (position.size() != 1)
              return extract->emitOpError(
                  "from a nested struct is not yet supported");
            define(extract.getResult(),
                   {scalarsOf(extract.getContainer())[position[0]]});
            return success();
          })
      .Case<LLVM::PoisonOp>([&](LLVM::PoisonOp poison) -> LogicalResult {
        FailureOr<SmallVector<Type>> types = scalarTypesOf(poison.getType());
        if (failed(types))
          return poison->emitOpError("of this type is not yet supported");
        SSAScalars scalars;
        for (Type type : *types)
          scalars.push_back(
              uniform({context.bv_val(0, type.getIntOrFloatBitWidth()),
                       context.bool_val(true)}));
        define(poison.getResult(), std::move(scalars));
        return success();
      })
      .Case<LLVM::InsertValueOp>(
          [&](LLVM::InsertValueOp insert) -> LogicalResult {
            ArrayRef<int64_t> position = insert.getPosition();
            if (position.size() != 1)
              return insert->emitOpError(
                  "into a nested struct is not yet supported");
            SSAScalars scalars = scalarsOf(insert.getContainer());
            scalars[position[0]] = scalarOf(insert.getValue());
            define(insert.getResult(), std::move(scalars));
            return success();
          })
      // Both source and target end with a return
      .Case<func::ReturnOp, LLVM::ReturnOp>([&](Operation *ret) {
        for (Value operand : ret->getOperands())
          llvm::append_range(results, scalarsOf(operand));
        return success();
      })
      .Default([&](Operation *other) {
        return other->emitOpError("is not yet supported by Eidolon");
      });
}

FailureOr<Evaluation>
FunctionEvaluator::evaluate(FunctionOpInterface function,
                            ArrayRef<WarpValue> arguments) {
  Region &body = function.getFunctionBody();
  if (!body.hasOneBlock())
    return emitError(function.getLoc(),
                     "only functions with precisely one block are supported");
  Block &block = body.front();
  assert(arguments.size() == block.getNumArguments() &&
         "expected one WarpValue per argument"); // Issue in Eidolon
  for (unsigned i = 0; i < arguments.size(); ++i)
    define(block.getArgument(i), {arguments[i]});
  for (Operation &op : block)
    if (failed(evaluateOp(&op)))
      return failure();
  return Evaluation{std::move(results), undefinedBehaviour};
}

FailureOr<Evaluation> eidolon::evaluateFunction(FunctionOpInterface function,
                                                const BitVector &activeLanes,
                                                ArrayRef<WarpValue> arguments,
                                                z3::context &context) {
  assert(activeLanes.any() && "expected at least one active lane");
  FunctionEvaluator evaluator(activeLanes, context);
  return evaluator.evaluate(function, arguments);
}
