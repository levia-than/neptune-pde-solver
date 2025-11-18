'''
Author: leviathan 670916484@qq.com
Date: 2025-11-16 16:36:36
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:43:19
FilePath: /neptune-pde-solver/neptune_pde_describer/src/nptdsl/backend/python_backend.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
# nptdsl/backend/python_backend.py
from dataclasses import dataclass
from typing import Callable, TypeAlias, List

import numpy as np

from ..field import Field1D
from ..stencil_ir import StencilProgram1D, StencilOp1D

StepFn: TypeAlias = Callable[[Field1D, float], None]


@dataclass
class PythonStencilKernel:
    program: StencilProgram1D

    @classmethod
    def from_program(cls, program: StencilProgram1D) -> "PythonStencilKernel":
        return cls(program=program)

    def make_step_fn(self) -> StepFn:
        field = self.program.field
        ops: List[StencilOp1D] = list(self.program.ops)

        def step(u: Field1D, dt: float) -> None:
            if u is not field:
                raise ValueError("Kernel bound to a different Field1D instance")

            vals = u.values
            old = vals.copy()
            rhs = np.zeros_like(old)
            n = old.shape[0]

            # 累加所有 stencil op 的贡献
            for op in ops:
                r = op.linear.radius
                coeffs = op.linear.coeffs
                scale = op.linear.scale
                for i in range(r, n - r):
                    acc = 0.0
                    for k in range(-r, r + 1):
                        acc += coeffs[k + r] * old[i + k]
                    rhs[i] += scale * acc

            # 显式 Euler 时间积分
            vals[1:-1] = old[1:-1] + dt * rhs[1:-1]

        return step
