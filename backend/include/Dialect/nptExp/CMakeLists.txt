set(LLVM_TARGET_DEFINITIONS nptExpDialect.td)

mlir_tablegen(nptExpDialect.h.inc -gen-dialect-decls)
mlir_tablegen(nptExpDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS nptExpOps.td)
mlir_tablegen(nptExpOps.h.inc -gen-op-decls)
mlir_tablegen(nptExpOps.cpp.inc -gen-op-defs)
mlir_tablegen(nptExpOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(nptExpOpsEnums.cpp.inc -gen-enum-defs)


add_custom_target(nptExpDialectOps
  DEPENDS
    nptExpDialect.h.inc
    nptExpDialect.cpp.inc
    nptExpOps.h.inc
    nptExpOps.cpp.inc
    nptExpOpsEnums.h.inc
    nptExpOpsEnums.cpp.inc
)
