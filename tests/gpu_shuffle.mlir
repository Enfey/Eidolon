// RUN: not eidolon %s | FileCheck %s
// RUN: eidolon --print-walk %s | FileCheck %s --check-prefix=WALK
// RUN: eidolon --print-lowered %s | FileCheck %s --check-prefix=NVVM

gpu.module @shuffles {
  // NVVM-LABEL: llvm.func @shuffle()
  // NVVM:       nvvm.shfl.sync bfly

  // The reason for this is as follows:
  // Result `5` = `%predu` of `@gpu_shuffle`. The source mandates that lane
  // `k` - `o` is in [0-22]. `o` = `4` thus this holds for lanes `4-22`.
  // However, the target semantics for nvvm.shfl.sync up as declared in the
  // PTX manual mandate that the `valid` result is true iff `k - o` >= `clamp`
  // where `clamp = warpSize - width`. For lanes `4-12`, the lane id - `o` does 
  // not exceed `clamp`, and thus reads are invalid.  As such, the result for
  // these values differs across these lanes, and so the target does not refine
  // the source. I can only assume that the lowering to the clamp in in-tree
  // MLIR is incorrect. Note the lack of argument lines amongst other differences.
  // CHECK-LABEL: @shuffle
  // CHECK-NEXT:  on lanes 0-22: the target does not refine the source
  // CHECK-NEXT:  result 5 (i1) can differ on lanes 4-12
  // CHECK-NEXT:  example for lane 4 (first recorded difference):
  // CHECK-NEXT:  source  lanes 0-3: false; lanes 4-22: true
  // CHECK-NEXT:  target  lanes 0-12: false; lanes 13-22: true

  // WALK-LABEL: @shuffle
  // WALK-NEXT:  source
  // WALK-NEXT:  %0 = constant 1.000000e+00 : f32
  // WALK:       %3, %4 = shuffle xor value=%0 offset=%1 width=%2
  // WALK:       return %3 %5 %7 %9 %4 %6 %8 %10
  // WALK-NEXT:  target
  // WALK-NEXT:  %[[#VALUE:]] = constant 1.000000e+00 : f32
  // WALK-NEXT:  %[[#OFFSET:]] = constant 4 : i32
  // WALK:       %[[#MASK:]] = lshr
  // WALK-NEXT:  %[[#CLAMP:]] = sub
  // WALK-NEXT:  %[[#XOR:]] = shfl.sync bfly mask=%[[#MASK]] value=%[[#VALUE]] offset=%[[#OFFSET]] clamp=%[[#CLAMP]] -> (value, valid)
  // WALK-NEXT:  extractvalue %[[#XOR]][0]
  func.func @shuffle() -> (f32, f32, f32, f32, i1, i1, i1, i1) {
    %arg0 = arith.constant 1.0 : f32
    %arg1 = arith.constant 4 : i32
    %arg2 = arith.constant 23 : i32
    %shfl, %pred = gpu.shuffle xor %arg0, %arg1, %arg2 : f32
    %shflu, %predu = gpu.shuffle up %arg0, %arg1, %arg2 : f32
    %shfld, %predd = gpu.shuffle down %arg0, %arg1, %arg2 : f32
    %shfli, %predi = gpu.shuffle idx %arg0, %arg1, %arg2 : f32

    func.return %shfl, %shflu, %shfld, %shfli, %pred, %predu, %predd, %predi
      : f32, f32, f32, f32, i1, i1, i1, i1
  }

  // The predicate remains unused and as such nothing can differ between the target and source.
  // Lanes `4-12` still refuse to read on the target but because all lanes hold the same argument
  // the shuffle result never changes regardless.
  // CHECK-LABEL: @shuffle_unused_pred
  // CHECK-NEXT:  on lanes 0-22: the target refines the source

  // WALK-LABEL:  @shuffle_unused_pred
  // WALK:        target
  // WALK:        shfl.sync bfly {{.*}} -> value
  func.func @shuffle_unused_pred() -> (f32, f32, f32, f32) {
    %arg0 = arith.constant 1.0 : f32
    %arg1 = arith.constant 4 : i32
    %arg2 = arith.constant 23 : i32
    %shfl, %pred = gpu.shuffle xor %arg0, %arg1, %arg2 : f32
    %shflu, %predu = gpu.shuffle up %arg0, %arg1, %arg2 : f32
    %shfld, %predd = gpu.shuffle down %arg0, %arg1, %arg2 : f32
    %shfli, %predi = gpu.shuffle idx %arg0, %arg1, %arg2 : f32

    func.return %shfl, %shflu, %shfld, %shfli : f32, f32, f32, f32
  }
}
