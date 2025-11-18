'''
Author: leviathan 670916484@qq.com
Date: 2025-11-16 16:36:29
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:43:10
FilePath: /neptune-pde-solver/neptune_pde_describer/src/nptdsl/backend/base.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
# nptdsl/backend/base.py
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Callable

from ..field import Field1D
from ..stencil_ir import StencilProgram1D


class BackendKind(Enum):
    PYTHON = "python"
    MLIR_AOT = "mlir_aot"
    MLIR_JIT = "mlir_jit"


class BackendKernel(Protocol):
    """
    后端产出的「已编译 kernel」统一接口。
    """
    def step(self, u: Field1D, dt: float) -> None:
        ...
