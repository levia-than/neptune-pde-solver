'''
Author: leviathan 670916484@qq.com
Date: 2025-11-14 20:44:44
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:55:25
FilePath: /neptune-pde-solver/neptune_pde_describer/tests/simple_test.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
import numpy as np
from nptdsl import (
    Grid1D, Field1D, Equation, d_t, d_xx,
    Method, BackendKind, TimeStepper,
)

g = Grid1D(nx=101, length=1.0)
u = Field1D("u", g)
alpha = 0.1

x = g.x
u.values[:] = np.sin(np.pi * x)
u.set_dirichlet(left=0.0, right=0.0)

eq = Equation(d_t(u), alpha * d_xx(u))

stepper = TimeStepper(eq, method=Method.FDM, backend=BackendKind.MLIR_AOT).build()
result = stepper.run(u, t_final=0.1, dt=1e-4)

print("Final u field after t=0.1s:")
print(result.values)
