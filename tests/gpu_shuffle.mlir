gpu.module @test_module_4 {
  // CHECK-LABEL: func @gpu_shuffle()
  func.func @gpu_shuffle() -> (f32, f32, f32, f32, i1, i1, i1, i1) {
    %arg0 = arith.constant 1.0 : f32
    %arg1 = arith.constant 4 : i32
    %arg2 = arith.constant 23 : i32
    %shfl, %pred = gpu.shuffle xor %arg0, %arg1, %arg2 : f32
    %shflu, %predu = gpu.shuffle up %arg0, %arg1, %arg2 : f32
    %shfld, %predd = gpu.shuffle down %arg0, %arg1, %arg2 : f32
    %shfli, %predi = gpu.shuffle idx %arg0, %arg1, %arg2 : f32

    func.return %shfl, %shflu, %shfld, %shfli, %pred, %predu, %predd, %predi
      : f32, f32,f32, f32, i1, i1, i1, i1
  }

  // CHECK-LABEL: func @gpu_shuffle_unused_pred()
  func.func @gpu_shuffle_unused_pred() -> (f32, f32, f32, f32) {
    %arg0 = arith.constant 1.0 : f32
    %arg1 = arith.constant 4 : i32
    %arg2 = arith.constant 23 : i32
    %shfl, %pred = gpu.shuffle xor %arg0, %arg1, %arg2 : f32
    %shflu, %predu = gpu.shuffle up %arg0, %arg1, %arg2 : f32
    %shfld, %predd = gpu.shuffle down %arg0, %arg1, %arg2 : f32
    %shfli, %predi = gpu.shuffle idx %arg0, %arg1, %arg2 : f32

    func.return %shfl, %shflu, %shfld, %shfli : f32, f32,f32, f32
  }
}