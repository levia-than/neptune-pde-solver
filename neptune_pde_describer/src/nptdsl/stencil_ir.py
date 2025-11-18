'''
Author: leviathan 670916484@qq.com
Date: 2025-11-16 16:42:20
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:51:30
FilePath: /neptune-pde-solver/neptune_pde_describer/src/nptdsl/stencil_ir.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
# nptdsl/stencil_ir.py
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
from .field import Field1D
from .pde_ir import TermKind

class StencilSpace(Enum):
    CELL = "cell"   # FVM: cell-center unknowns
    NODE = "node"   # FDM/FEM: node-based unknowns


@dataclass
class LinearStencil1D:
    """
    线性 stencil：适合 FDM diffusion 等。
    对某个 i，有：
        L(u)_i = scale * sum_k coeffs[k] * u[i + (k-radius)]
    """
    radius: int
    coeffs: np.ndarray  # 长度 = 2*radius+1
    scale: float        # 全局缩放（比如 alpha/dx^2）


@dataclass
class StencilOp1D:
    """
    离散后的单个空间算子：
      - 目前只存线性 stencil（FDM MVP）
      - 将来可以加 kind / flux_type 等扩展 FVM
    """
    field: Field1D
    space: StencilSpace          # NODE / CELL
    term_kind: TermKind          # 来自 PDE IR：DIFFUSION / CONVECTION / ...
    linear: LinearStencil1D      # 目前只支持线性 FDM
    # flux_kind: Optional[FluxKind] = None   # 标记 Rusanov / Godunov 等


@dataclass
class StencilProgram1D:
    """
    针对某个 field 的全部一维 stencil 操作。
    """
    field: Field1D
    ops: List[StencilOp1D]
