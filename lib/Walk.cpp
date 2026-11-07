//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the walk. It is a switch over op kinds yielding unique
//  ids for newly encountered SSA values in ops recognised by Eidolon.
///
//===----------------------------------------------------------------------===//

#include "../include/eidolon/Walk.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;

namespace {
class FunctionWalker {
public:
  explicit FunctionWalker(raw_ostream &os) : os(os) {}

  /// Walks the function's block in-order
  LogicalResult walk(FunctionOpInterface function);

private:
  /// Numbers SSA values in order of appearance (temp)
  using ValueIDs = DenseMap<Value, unsigned>;

  /// Gives a value the next available id, and returns that id, "%N"
  std::string define(Value value);

  /// Starts a line of pseudocode, indented to sit under its function's
  /// heading, and returns the stream for the rest of the line
  raw_ostream &startLine() { return os.indent(4); }

  /// Returns the id of an SSA value registered earlier in the walk
  std::string idOf(Value value);

  LogicalResult walkOp(Operation &op);

  raw_ostream &os;
  ValueIDs valueIDs;
};
} // namespace

std::string FunctionWalker::define(Value value) {
  unsigned number = valueIDs.size();
  valueIDs[value] = number;
  return "%" + std::to_string(number);
}

std::string FunctionWalker::idOf(Value value) {
  assert(valueIDs.contains(value) && "value used before the walk defined it");
  return "%" + std::to_string(valueIDs.lookup(value));
}

LogicalResult FunctionWalker::walkOp(Operation &op) {
  return TypeSwitch<Operation *, LogicalResult>(&op)
      // Source ops
      .Case<arith::ConstantOp>([&](arith::ConstantOp constant) {
        startLine() << define(constant.getResult()) << " = constant "
                    << constant.getValue() << "\n";
        return success();
      })
      .Case<gpu::ShuffleOp>([&](gpu::ShuffleOp shuffle) {
        std::string result = define(shuffle.getShuffleResult());
        std::string valid = define(shuffle.getValid());
        startLine() << result << ", " << valid << " = shuffle "
                    << gpu::stringifyShuffleMode(shuffle.getMode())
                    << " value=" << idOf(shuffle.getValue())
                    << " offset=" << idOf(shuffle.getOffset())
                    << " width=" << idOf(shuffle.getWidth()) << "\n";
        return success();
      })
      // Target ops
      .Case<LLVM::ConstantOp>([&](LLVM::ConstantOp constant) {
        startLine() << define(constant.getResult()) << " = constant "
                    << constant.getValue() << "\n";
        return success();
      })
      .Case<LLVM::SubOp>([&](LLVM::SubOp sub) {
        startLine() << define(sub.getResult()) << " = sub "
                    << idOf(sub.getLhs()) << ", " << idOf(sub.getRhs()) << "\n";
        return success();
      })
      .Case<LLVM::LShrOp>([&](LLVM::LShrOp shift) {
        startLine() << define(shift.getResult()) << " = lshr "
                    << idOf(shift.getLhs()) << ", " << idOf(shift.getRhs())
                    << "\n";
        return success();
      })
      .Case<NVVM::ShflOp>([&](NVVM::ShflOp shuffle) {
        startLine() << define(shuffle.getResult()) << " = shfl.sync "
                    << NVVM::stringifyShflKind(shuffle.getKind())
                    << " mask=" << idOf(shuffle.getThreadMask())
                    << " value=" << idOf(shuffle.getVal())
                    << " offset=" << idOf(shuffle.getOffset())
                    << " clamp=" << idOf(shuffle.getMaskAndClamp())
                    << (shuffle.getReturnValueAndIsValidAttr()
                            ? " -> (value, valid)"
                            : " -> value")
                    << "\n";
        return success();
      })
      .Case<LLVM::ExtractValueOp>([&](LLVM::ExtractValueOp extract) {
        startLine() << define(extract.getResult()) << " = extractvalue "
                    << idOf(extract.getContainer());
        for (int64_t index : extract.getPosition())
          os << "[" << index << "]";
        os << "\n";
        return success();
      })
      .Case<LLVM::PoisonOp>([&](LLVM::PoisonOp poison) {
        startLine() << define(poison.getResult()) << " = poison\n";
        return success();
      })
      .Case<LLVM::InsertValueOp>([&](LLVM::InsertValueOp insert) {
        startLine() << define(insert.getResult()) << " = insertvalue "
                    << idOf(insert.getValue()) << " into "
                    << idOf(insert.getContainer());
        for (int64_t index : insert.getPosition())
          os << "[" << index << "]";
        os << "\n";
        return success();
      })
      // Naturally both source and target end with a return
      .Case<func::ReturnOp, LLVM::ReturnOp>([&](Operation *ret) {
        startLine() << "return";
        for (Value operand : ret->getOperands())
          os << " " << idOf(operand);
        os << "\n";
        return success();
      })
      .Default([&](Operation *other) {
        return other->emitOpError("is not yet supported by Eidolon");
      });
}

LogicalResult FunctionWalker::walk(FunctionOpInterface function) {
  Region &body = function.getFunctionBody();
  if (!body.hasOneBlock())
    return emitError(function.getLoc(),
                     "only functions with precisely one block are supported");
  Block &block = body.front();
  for (BlockArgument argument : block.getArguments())
    startLine() << define(argument) << " = argument\n";
  for (Operation &op : block)
    if (failed(walkOp(op)))
      return failure();
  return success();
}

LogicalResult eidolon::walkFunction(FunctionOpInterface function,
                                    raw_ostream &os) {
  FunctionWalker walker(os);
  return walker.walk(function);
}
