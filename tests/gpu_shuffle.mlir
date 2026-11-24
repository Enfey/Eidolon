// RUN: eidolon %s | FileCheck %s --check-prefix=WALK
// RUN: eidolon --print-lowered %s | FileCheck %s --check-prefix=NVVM

// Although these tests are ephmeral, in hindsight removing the dialect names from the pseudocode we generate when walking was not a good idea

gpu.module @test_module_4 {
  // NVVM-LABEL: llvm.func @gpu_shuffle()
  // NVVM:       nvvm.shfl.sync bfly

  // WALK-LABEL: @gpu_shuffle
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
  func.func @gpu_shuffle() -> (f32, f32, f32, f32, i1, i1, i1, i1) {
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

  // WALK-LABEL: @gpu_shuffle_unused_pred
  // WALK:       target
  // WALK:       shfl.sync bfly {{.*}} -> value
  func.func @gpu_shuffle_unused_pred() -> (f32, f32, f32, f32) {
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
