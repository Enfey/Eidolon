add_mlir_library(EidolonLib
  Lowering.cpp
  Walk.cpp
  Evaluate.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRGPUDialect
  MLIRGPUToNVVMTransforms
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRPass
)
