'''
Author: leviathan 670916484@qq.com
Date: 2025-11-16 16:36:42
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:43:23
FilePath: /neptune-pde-solver/neptune_pde_describer/src/nptdsl/backend/mlir_backend.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
# nptdsl/backend/mlir_backend.py
from dataclasses import dataclass
from typing import Callable, TypeAlias, Optional

from ..field import Field1D
from ..stencil_ir import StencilProgram1D
from .base import BackendKind

StepFn: TypeAlias = Callable[[Field1D, float], None]

# 尝试加载原生 MLIR 绑定（未来用 pybind11 暴露 C++ NeptuneModuleBuilder）。
try:
    from nptdsl import _neptune_mlir as _native_mlir  # type: ignore
except Exception:  # pragma: no cover - 缺省为未构建状态
    _native_mlir = None  # noqa: N816


@dataclass
class MlirStencilKernel:
    program: StencilProgram1D
    kind: BackendKind  # MLIR_AOT / MLIR_JIT

    @classmethod
    def from_program(cls, program: StencilProgram1D, kind: BackendKind) -> "MlirStencilKernel":
        if kind not in (BackendKind.MLIR_AOT, BackendKind.MLIR_JIT):
            raise ValueError(f"Invalid MLIR backend kind: {kind}")
        return cls(program=program, kind=kind)

    def make_step_fn(self) -> StepFn:
        def step(_u: Field1D, _dt: float) -> None:
            # 还未完成 JIT/AOT，但先尝试生成 NeptuneIR 模块文本，便于调试。
            if _native_mlir is None:
                raise NotImplementedError(
                    "MLIR backend is not built yet (missing _neptune_mlir extension)"
                )

            # 只构造一次 Module 文本（包括 stencil_step 函数），先打印出来。
            mlir_module: Optional[str] = getattr(self, "_cached_mlir", None)
            if mlir_module is None:
                mlir_module = _native_mlir.build_stencil_module(self.program)
                self._cached_mlir = mlir_module
                print(mlir_module)

            raise NotImplementedError(
                f"MLIR backend ({self.kind.value}) codegen/runtime not wired yet"
            )

        return step
