message("building nptImpl.")
add_mlir_dialect(NptImplOps npt_impl)
add_mlir_doc(NptImplOps NptImplDialect Dialects/ -gen-dialect-doc)
