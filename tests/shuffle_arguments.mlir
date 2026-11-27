// RUN: not eidolon %s | FileCheck %s
// RUN: not eidolon %s 2>&1 | FileCheck %s --check-prefix=ERROR

gpu.module @shuffles {
  // CHECK-LABEL: @shuffle_up_w_23_vo_arg
  // CHECK-NEXT:  on lanes 0-22: the target does not refine the source
  // CHECK-NEXT:  result 0 (f32) can differ on lanes 1-22
  // CHECK-NEXT:  example for lane 1 (first recorded difference):
  // CHECK-NEXT:  argument 0 (f32) {{.*}}
  // CHECK-NEXT:  argument 1 (i32) {{.*}}
  // CHECK-NEXT:  source {{.*}}
  // CHECK-NEXT:  target {{.*}}
  // CHECK-NEXT:  result 1 (i1) can differ on lanes 0-22
  // CHECK-NEXT:  example for lane 0 (first recorded difference):
  func.func @shuffle_up_w_23_vo_arg(%x: f32, %offset: i32) -> (f32, i1) {
    %width = arith.constant 23 : i32
    %result, %valid = gpu.shuffle up %x, %offset, %width : f32
    return %result, %valid : f32, i1
  }

  // CHECK-LABEL: @shuffle_down_w_23_vo_arg
  // CHECK-NEXT:  on lanes 0-22: the target refines the source
  func.func @shuffle_down_w_23_vo_arg(%x: f32, %offset: i32) -> (f32, i1) {
    %width = arith.constant 23 : i32
    %result, %valid = gpu.shuffle down %x, %offset, %width : f32
    return %result, %valid : f32, i1
  }

  // The lowering passes 32 - width as the clamp of `up`'s reads as such the target
  // reads are invalid for lanes 0-15 regardless of offset, and the results differ 
  // as the read does not take place on the target (the values are not uniform).
  // Note that the value does not differ for lane `0` as the offset cannot take a
  // value where the lane contents would differ (0) or where the lane read would occur
  // in the source.
  // CHECK-LABEL: @shuffle_up_w_16_vo_arg
  // CHECK-NEXT:  on lanes 0-15: the target does not refine the source
  // CHECK-NEXT:  result 0 (f32) can differ on lanes 1-15
  // CHECK-NEXT:  example for lane 1 (first recorded difference):
  // CHECK-NEXT:  argument 0 (f32) {{.*}}
  // CHECK-NEXT:  argument 1 (i32) {{.*}}
  // CHECK-NEXT:  source {{.*}}
  // CHECK-NEXT:  target {{.*}}
  // CHECK-NEXT:  result 1 (i1) can differ on lanes 0-15
  // CHECK-NEXT:  example for lane 0 (first recorded difference):
  func.func @shuffle_up_w_16_vo_arg(%x: f32, %offset: i32) -> (f32, i1) {
    %width = arith.constant 16 : i32
    %result, %valid = gpu.shuffle up %x, %offset, %width : f32
    return %result, %valid : f32, i1
  }

  // CHECK-LABEL: @shuffle_up_w_32_vo_arg
  // CHECK-NEXT:  on lanes 0-31: the target refines the source
  func.func @shuffle_up_w_32_vo_arg(%x: f32, %offset: i32) -> (f32, i1) {
    %width = arith.constant 32 : i32
    %result, %valid = gpu.shuffle up %x, %offset, %width : f32
    return %result, %valid : f32, i1
  }

  // TBC. Once we make the width an unknown that is uniform across lanes
  // this test should fail. 
  // CHECK-LABEL:  @shuffle_up_vow_arg
  // ERROR: error: 'gpu.shuffle' op needs a constant width (temporary)
  func.func @shuffle_up_vow_arg(%x: f32, %offset: i32, %width: i32) -> (f32, i1) {
    %result, %valid = gpu.shuffle down %x, %offset, %width : f32
    return %result, %valid : f32, i1
  }
}