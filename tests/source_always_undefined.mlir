// RUN: not eidolon %s | FileCheck %s

gpu.module @shuffles {
  // For lane `k`, if `k` + `%offset` exceeds `%width`, the value is poison and valid is `false`.
  // There will be no remaining input to check refinement over, as `k + %offset` always exceeds `width`.

  // CHECK-LABEL: @shuffle_down_32_o_23_w_v_arg
  // CHECK-NEXT:  on lanes 0-22: the source has undefined behaviour on every input, so there is nothing to check
  func.func @shuffle_down_32_o_23_w_v_arg(%x: f32) -> f32 {
    %offset = arith.constant 32 : i32
    %width = arith.constant 23 : i32
    %result, %valid = gpu.shuffle down %x, %offset, %width : f32
    return %result : f32
  }
}