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