// RUN: not eidolon %s 2>&1 | FileCheck %s

// CHECK: error: 'arith.ceildivsi' op is not yet supported by Eidolon
gpu.module @unsupported {
  func.func @divide(%a: i32, %b: i32) -> i32 {
    %quotient = arith.ceildivsi %a, %b : i32
    func.return %quotient : i32
  }
}