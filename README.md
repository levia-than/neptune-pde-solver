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