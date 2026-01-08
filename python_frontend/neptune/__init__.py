'''
Author: leviathan 670916484@qq.com
Date: 2025-11-14 17:06:27
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-29 16:54:31
FilePath: /neptune-pde-solver/python_frontend/neptune/__init__.py
Description: Neptune PDE Solver Python Frontend
Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''

# 1. 暴露核心上下文
from .core import GlobalContext as Context
# 方便直接获取全局编译器实例 (如果需要的话)
from .core import get_compiler 

# 2. 暴露 IR 表达式对象
from .expr import Expr

# 3. 暴露 DSL 装饰器和指令
from .dsl import (
    apply,
    stencil,
    linear_op_def,
    assemble_matrix,
    solve_linear,
)

# 5. [New] 暴露 JIT 编译接口 (让用户能一键获得 .so)
from .backend import jit_compile

from .jit import jit_class

__all__ = [
    "Context",
    "get_compiler",
    "Expr",
    "apply",
    "stencil",
    "linear_op_def",
    "assemble_matrix",
    "solve_linear",
    "jit_compile",
    "jit_class"
]

