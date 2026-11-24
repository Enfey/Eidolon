// RUN: not eidolon %s 2>&1 | FileCheck %s

// CHECK: error: nothing to check: no func.func inside a gpu.module
func.func @outside_gpu_module(%a: i32) -> i32 {
  func.return %a : i32
}