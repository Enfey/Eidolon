cmake_minimum_required(VERSION 3.31)
project(Eidolon LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

# Arch does not ship a static archive of libzstd yet LLVM's exported targets ask for one.
add_library(zstd::libzstd_static UNKNOWN IMPORTED)
set_target_properties(zstd::libzstd_static PROPERTIES IMPORTED_LOCATION /usr/lib/libzstd.so)

find_package(MLIR REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions) # Compile Eidolon with exactly the same compiler flags LLVm was built with (-fno-rtti -fno-exceptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(Z3 CONFIG REQUIRED)

add_subdirectory(lib)
add_subdirectory(tools/eidolon)
