'''
Author: leviathan 670916484@qq.com
Date: 2025-11-14 17:06:27
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:54:31
FilePath: /neptune-pde-solver/neptune_pde_describer/src/nptdsl/__init__.py
Description:  

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
# src/nptdsl/__init__.py

from .grid import Grid1D
from .field import Field1D
from .equation import Equation, d_t, d_xx

# PDE IR 层（给需要的人用，不是必须导出全部）
from .pde_ir import (
    Method,
    PDEDescription,
    TermKind,
    TimeDescriptor,
    SpatialTerm,
    describe_equation,
)

# 离散配置 / stencil IR（如果你觉得现在不想暴露这么多，可以先不 export）
from .discretization import DiscretizationConfig, TimeScheme, discretize_1d
from .stencil_ir import (
    StencilSpace,
    LinearStencil1D,
    StencilOp1D,
    StencilProgram1D,
)

# 后端选择 + 时间推进
from .backend.base import BackendKind
from .timestepping import TimeStepper


__all__ = [
    # DSL
    "Grid1D",
    "Field1D",
    "Equation",
    "d_t",
    "d_xx",

    # PDE IR
    "Method",
    "PDEDescription",
    "TermKind",
    "TimeDescriptor",
    "SpatialTerm",
    "describe_equation",

    # Discretization / Stencil IR
    "DiscretizationConfig",
    "TimeScheme",
    "discretize_1d",
    "StencilSpace",
    "LinearStencil1D",
    "StencilOp1D",
    "StencilProgram1D",

    # Backend / Runtime
    "BackendKind",
    "TimeStepper",
]
