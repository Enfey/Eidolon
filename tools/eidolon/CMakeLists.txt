add_llvm_executable(eidolon eidolon.cpp)

target_link_libraries(eidolon PRIVATE
  EidolonLib
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  z3::libz3
)
