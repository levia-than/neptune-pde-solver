'''
Author: leviathan 670916484@qq.com
Date: 2025-11-16 16:36:09
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:43:32
FilePath: /neptune-pde-solver/neptune_pde_describer/src/nptdsl/timestepping.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
# nptdsl/timestepping.py
from dataclasses import dataclass
from typing import Optional, Callable, TypeAlias

from .equation import Equation
from .field import Field1D
from .pde_ir import Method, TimeOrder, PDEDescription, describe_equation
from .discretization import DiscretizationConfig, discretize_1d
from .stencil_ir import StencilProgram1D
from .backend.base import BackendKind
from .backend.python_backend import PythonStencilKernel
from .backend.mlir_backend import MlirStencilKernel

StepFn: TypeAlias = Callable[[Field1D, float], None]


@dataclass
class TimeStepper:
    equation: Equation
    method: Method = Method.FDM
    backend: BackendKind = BackendKind.PYTHON
    discretization: Optional[DiscretizationConfig] = None

    _desc: Optional[PDEDescription] = None
    _program: Optional[StencilProgram1D] = None
    _step_fn: Optional[StepFn] = None
    _target_field: Optional[Field1D] = None

    def build(self) -> "TimeStepper":
        # 1) PDE IR
        desc = describe_equation(self.equation, self.method)
        if desc.time.order is not TimeOrder.FIRST:
            raise ValueError("MVP only supports first-order time derivative d_t(u)")
        self._desc = desc

        # 2) 离散 IR (1D)
        cfg = self.discretization or DiscretizationConfig(method=self.method)
        program = discretize_1d(desc, cfg)
        self._program = program
        self._target_field = program.field

        # 3) backend kernel
        if self.backend is BackendKind.PYTHON:
            kernel = PythonStencilKernel.from_program(program)
            self._step_fn = kernel.make_step_fn()
        elif self.backend in (BackendKind.MLIR_AOT, BackendKind.MLIR_JIT):
            kernel = MlirStencilKernel.from_program(program, self.backend)
            self._step_fn = kernel.make_step_fn()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        return self

    def step(self, u: Field1D, dt: float) -> None:
        if self._step_fn is None:
            self.build()
        self._ensure_field_matches(u)
        self._step_fn(u, dt)  # type: ignore[arg-type]

    def run(self, u: Field1D, t_final: float, dt: float) -> Field1D:
        if self._step_fn is None:
            self.build()
        self._ensure_field_matches(u)
        n_steps = int(round(t_final / dt))
        for _ in range(n_steps):
            self.step(u, dt)
        return u

    def _ensure_field_matches(self, u: Field1D) -> None:
        if self._target_field is None:
            return
        if self._target_field is not u:
            raise ValueError(
                "TimeStepper was built for a different Field1D instance; "
                "MVP assumes a single bound field."
            )
