message("building NeptuneIR.")
add_mlir_dialect(NeptuneIROps neptune_ir)
add_mlir_doc(NeptuneIROps NeptuneIRDialect Dialects/ -gen-op-doc)
