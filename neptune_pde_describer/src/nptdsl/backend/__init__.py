'''
Author: leviathan 670916484@qq.com
Date: 2025-11-16 16:36:24
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:54:42
FilePath: /neptune-pde-solver/neptune_pde_describer/src/nptdsl/backend/__init__.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
# src/nptdsl/backend/__init__.py

from .base import BackendKind
from .python_backend import PythonStencilKernel
from .mlir_backend import MlirStencilKernel

__all__ = [
    "BackendKind",
    "PythonStencilKernel",
    "MlirStencilKernel",
]
