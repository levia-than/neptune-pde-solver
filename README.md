<!--
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-09-08 20:34:11
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-09 10:33:05
 * @FilePath: /neptune-pde-solver/README.md
 * @Description: 
 * 
 * Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
-->
# Neptune pde solver

```bash
git submodule update --init
bash scripts/build.sh
```

```bash
bash scripts/build.sh -c
```
this would clean up build directory.

## basic arch
`lib` directory would contain all mlir-related library.
`src` directory would export some opt tools for the sake of testing.
`include` would contain all tablegen files.

## python front-end bridge
`lib/Utils` builds `MLIRNeptuneIRBuilder` as a shared library that exports
`mlir::Neptune::NeptuneIR::NeptuneIRBuilder`.  Pybind11 modules can link
against this `.so`, include `Utils/NeptuneIRBuilder.h`, and directly call
helpers such as `createFieldRef`/`createEvaluate` to construct NeptuneIR from
Python.  Install step drops the shared library under
`build/project-build/lib`, so a Python extension can dlopen it at runtime or be
linked statically if desired.

## basic usage
```bash
./build/project-build/bin/neptune-opt ./docs/test.mlir --symbolic-simplify --lower-evaluate-to-real-compute --canonicalize --cse --convert-scf-to-cf --convert-func-to-llvm --expand-strided-metadata --finalize-memref-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts -o lower.mlir
```

```bash
./build/llvm-install/bin/mlir-translate --mlir-to-llvmir lower.mlir -o lower.ll
```

```bash
./build/llvm-install/bin/llc -filetype=obj lower.ll -o lower.o
```

## aot helpers
Two helper CLIs live under `src/`:

1. `neptune-compile` lowers a NeptuneIR module all the way to LLVM, emits
   `after_llvm.ll`, `kernel.o`, and links a shared library exporting the
   fixed `run_kernel` ABI from `Codegen/AOTABI.h`.
2. `neptune-run` dlopens the produced library, prepares a tiny tensor
   descriptor, calls `run_kernel`, and prints the first few outputs.

For Python runtime validation you can use:

Python/pybind helpers are currently disabled in the build; only the C++/MLIR
pipeline is configured by default.
