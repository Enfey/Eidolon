#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, /* Would expect a flag otherwise*/
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::Required);

int main(int argc, char **argv) {
    llvm::InitLLVM init(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv, "Eidolon\n");
    
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    mlir::MLIRContext context(registry);

    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler diagnosticHandler(sourceMgr, &context);

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, sourceMgr, &context);
    if (!module)
        return 1;

  module->print(llvm::outs()); /* Where OwningOpRef is essentially smart-pointer wrapper, and -> is overloaded to yield the underlying op (and the print method) */
  llvm::outs() << "\n"; /* Is implemented as a singleton for llvm::raw_fd_ostream*/
  return 0;
}
