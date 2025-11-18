'''
Author: leviathan 670916484@qq.com
Date: 2025-11-14 17:05:57
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-14 17:09:05
FilePath: /neptune-pde-solver/neptune_pde_describer/src/grid.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Grid1D:
    """
    最简单的一维均匀网格。
    nx: 网格点数量（包含边界点）
    length: 区间长度 [0, length]
    """
    nx: int
    length: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 3:
            raise ValueError("Grid1D requires nx >= 3 for FDM stencils")

    @property
    def dx(self) -> float:
        return self.length / (self.nx - 1)

    @property
    def x(self) -> np.ndarray:
        """网格点坐标（含两端点）."""
        return np.linspace(0.0, self.length, self.nx, dtype=float)