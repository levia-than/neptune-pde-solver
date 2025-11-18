'''
Author: leviathan 670916484@qq.com
Date: 2025-11-16 16:41:42
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:43:00
FilePath: /neptune-pde-solver/neptune_pde_describer/src/nptdsl/discretization.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
# nptdsl/discretization.py
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .pde_ir import PDEDescription, Method, TermKind
from .stencil_ir import (
    StencilProgram1D,
    StencilOp1D,
    StencilSpace,
    LinearStencil1D,
)


class TimeScheme(Enum):
    EXPLICIT_EULER = "explicit_euler"
    # 将来可以加 RK2, RK4, CN, BDF...


@dataclass
class DiscretizationConfig:
    method: Method = Method.FDM
    time_scheme: TimeScheme = TimeScheme.EXPLICIT_EULER
    space_order: int = 2   # 目前只用到 2
    # 将来可以加：convection_scheme, flux_type 等


def discretize_1d(desc: PDEDescription, config: DiscretizationConfig) -> StencilProgram1D:
    """
    MVP：只支持 1D + FDM + 二阶 diffusion。
    以后你可以在这里加 FVM 分支、对流项等。
    """
    if desc.method is not Method.FDM:
        raise ValueError("MVP only supports Method.FDM in discretize_1d")

    field = desc.time.field
    ops: list[StencilOp1D] = []

    for term in desc.terms:
        if term.kind is TermKind.DIFFUSION:
            dx = field.grid.dx
            radius = 1
            coeffs = np.array([1.0, -2.0, 1.0], dtype=float)
            scale = term.coefficient / (dx * dx)
            linear = LinearStencil1D(radius=radius, coeffs=coeffs, scale=scale)
            op = StencilOp1D(
                field=field,
                space=StencilSpace.NODE,
                term_kind=TermKind.DIFFUSION,
                linear=linear,
            )
            ops.append(op)
        else:
            raise ValueError(f"Term kind {term.kind} not yet supported in MVP")

    if not ops:
        raise ValueError("No spatial terms found for discretization")

    return StencilProgram1D(field=field, ops=ops)
