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
from typing import Callable, TypeAlias

from ..field import Field1D
from ..stencil_ir import StencilProgram1D
from .base import BackendKind

StepFn: TypeAlias = Callable[[Field1D, float], None]


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
        # TODO: 这里以后接入 MLIR pipeline
        def step(_u: Field1D, _dt: float) -> None:
            raise NotImplementedError(
                f"MLIR backend ({self.kind.value}) is not implemented yet"
            )

        return step
