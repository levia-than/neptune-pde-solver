set(LLVM_TARGET_DEFINITIONS nptImplDialect.td)

mlir_tablegen(nptImplDialect.h.inc -gen-dialect-decls)
mlir_tablegen(nptImplDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS nptImplOps.td)
mlir_tablegen(nptImplOps.h.inc -gen-op-decls)
mlir_tablegen(nptImplOps.cpp.inc -gen-op-defs)
mlir_tablegen(nptImplOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(nptImplOpsEnums.cpp.inc -gen-enum-defs)


add_custom_target(nptExpDialectOps
  DEPENDS
    nptImplDialect.h.inc
    nptImplDialect.cpp.inc
    nptImplOps.h.inc
    nptImplOps.cpp.inc
    nptImplOpsEnums.h.inc
    nptImplOpsEnums.cpp.inc
)
