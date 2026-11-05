cmake_minimum_required(VERSION 3.31)
project(Eidolon LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

# Arch does not ship a static archive of libzstd yet LLVM's exported targets ask for one.
add_library(zstd::libzstd_static UNKNOWN IMPORTED)
set_target_properties(zstd::libzstd_static PROPERTIES IMPORTED_LOCATION /usr/lib/libzstd.so)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions) # Compile Eidolon with exactly the same compiler flags LLVm was built with (-fno-rtti -fno-exceptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

find_package(Z3 CONFIG REQUIRED)

add_llvm_executable(Eidolon src/main.cpp)
target_link_libraries(Eidolon PRIVATE
  MLIRParser MLIRPass MLIRGPUDialect MLIRFuncDialect MLIRArithDialect
  MLIRLLVMDialect MLIRNVVMDialect MLIRGPUToNVVMTransforms
  MLIRRegisterAllDialects MLIRRegisterAllExtensions
  z3::libz3)

