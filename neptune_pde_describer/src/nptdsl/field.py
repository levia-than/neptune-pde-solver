'''
Author: leviathan 670916484@qq.com
Date: 2025-11-14 17:06:07
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-14 17:09:54
FilePath: /neptune-pde-solver/neptune_pde_describer/src/field.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
from typing import Optional
import numpy as np

from .grid import Grid1D


class Field1D:
    """
    在 Grid1D 上的标量场。内部就是一个 1D numpy array。
    先只支持 Dirichlet 边界，方便 1D FDM。
    """

    def __init__(self, name: str, grid: Grid1D, dtype=float) -> None:
        self.name = name
        self.grid = grid
        self.data = np.zeros(grid.nx, dtype=dtype)

    # 一些方便的别名
    @property
    def values(self) -> np.ndarray:
        return self.data

    @values.setter
    def values(self, arr: np.ndarray) -> None:
        if arr.shape != self.data.shape:
            raise ValueError(f"shape mismatch: {arr.shape} != {self.data.shape}")
        self.data[...] = arr

    def copy(self, name: Optional[str] = None) -> "Field1D":
        other = Field1D(name or self.name, self.grid, self.data.dtype)
        other.data[...] = self.data
        return other

    # 极简 Dirichlet 边界设置（只是改 data[0] / data[-1]）
    def set_dirichlet(self, left: Optional[float] = None, right: Optional[float] = None) -> None:
        if left is not None:
            self.data[0] = left
        if right is not None:
            self.data[-1] = right

    def __repr__(self) -> str:  # 调试方便一点
        return f"Field1D(name={self.name!r}, nx={self.grid.nx})"
