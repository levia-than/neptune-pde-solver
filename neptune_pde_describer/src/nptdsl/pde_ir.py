'''
Author: leviathan 670916484@qq.com
Date: 2025-11-16 16:36:01
LastEditors: leviathan 670916484@qq.com
LastEditTime: 2025-11-16 16:50:40
FilePath: /neptune-pde-solver/neptune_pde_describer/src/nptdsl/pde_ir.py
Description: 

Copyright (c) 2025 by leviathan, All Rights Reserved. 
'''
# nptdsl/pde_ir.py
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum
from typing import List
from .field import Field1D
from .equation import Equation, Expr

class Method(Enum):
    FDM = "fdm"
    FVM = "fvm"
    FEM = "fem"

class TimeOrder(Enum):
    FIRST = 1
    SECOND = 2

class TermKind(Enum):
    DIFFUSION = "diffusion"
    CONVECTION = "convection"
    SOURCE = "source"
    MASS = "mass"

@dataclass
class TimeDescriptor:
    field: Field1D
    order: TimeOrder
    coefficient: float = 1.0  # 预留给 a * d_t(u)

@dataclass
class SpatialTerm:
    kind: TermKind
    field: Field1D
    operator: Expr
    coefficient: float = 1.0  # 包含符号

@dataclass
class PDEDescription:
    method: Method
    time: TimeDescriptor
    terms: List[SpatialTerm]


def _parse_time_lhs(lhs: Expr) -> TimeDescriptor:
    """
    解析方程左边的时间项，目前只支持：
      - d_t(u)
      - d_t2(u)（预留）
    将来可以扩展：
      - a * d_t(u)
    """
    if lhs.op == "dt":
        (inner,) = lhs.args
        if inner.op != "var" or not isinstance(inner.args[0], Field1D):
            raise ValueError("dt(...) must be applied to a Field1D")
        field = inner.args[0]
        return TimeDescriptor(field=field, order=TimeOrder.FIRST, coefficient=1.0)

    if lhs.op == "dt2":
        (inner,) = lhs.args
        if inner.op != "var" or not isinstance(inner.args[0], Field1D):
            raise ValueError("dt2(...) must be applied to a Field1D")
        field = inner.args[0]
        return TimeDescriptor(field=field, order=TimeOrder.SECOND, coefficient=1.0)

    raise ValueError("Currently only dt(u) or dt2(u) are allowed on the LHS")


def _split_rhs_terms(expr: Expr) -> List[Tuple[float, Expr]]:
    """
    把 RHS 的 Expr 展开成若干项：
        add(a, b)      -> [ (+1, a), (+1, b) ]
        sub(a, b)      -> [ (+1, a), (-1, b) ]
        mul(-1, term)  -> [ (-1, term) ]
    其它情况当成一个整体项返回。
    """
    terms: List[Tuple[float, Expr]] = []

    def visit(node: Expr, sign: float) -> None:
        if node.op == "add":
            a, b = node.args
            visit(a, sign)
            visit(b, sign)
        elif node.op == "sub":
            a, b = node.args
            visit(a, sign)
            visit(b, -sign)
        elif node.op == "mul":
            a, b = node.args
            # 提取常数因子
            if a.op == "const":
                (c,) = a.args
                visit(b, sign * float(c))
            elif b.op == "const":
                (c,) = b.args
                visit(a, sign * float(c))
            else:
                terms.append((sign, node))
        else:
            terms.append((sign, node))

    visit(expr, +1.0)
    return terms

def _classify_term_for_fdm(expr: Expr, sign: float) -> SpatialTerm:
    """
    FDM 模式下的项分类。
    当前只支持：
      - dxx(u)
      - alpha * dxx(u) / dxx(u) * alpha
    一律归为 DIFFUSION。
    """
    # 情况 1：dxx(u)
    if expr.op == "dxx":
        (inner,) = expr.args
        if inner.op == "var" and isinstance(inner.args[0], Field1D):
            field = inner.args[0]
            return SpatialTerm(
                kind=TermKind.DIFFUSION,
                field=field,
                operator=expr,
                coefficient=sign * 1.0,
            )
        raise ValueError("dxx(...) must be applied to a Field1D")

    # 情况 2：mul(const, dxx(u)) 或 mul(dxx(u), const)
    if expr.op == "mul":
        a, b = expr.args

        # const * dxx(u)
        if a.op == "const" and b.op == "dxx":
            (c,) = a.args
            (inner,) = b.args
            if inner.op == "var" and isinstance(inner.args[0], Field1D):
                field = inner.args[0]
                return SpatialTerm(
                    kind=TermKind.DIFFUSION,
                    field=field,
                    operator=b,
                    coefficient=sign * float(c),
                )
            raise ValueError("dxx(...) must be applied to a Field1D")

        # dxx(u) * const
        if b.op == "const" and a.op == "dxx":
            (c,) = b.args
            (inner,) = a.args
            if inner.op == "var" and isinstance(inner.args[0], Field1D):
                field = inner.args[0]
                return SpatialTerm(
                    kind=TermKind.DIFFUSION,
                    field=field,
                    operator=a,
                    coefficient=sign * float(c),
                )
            raise ValueError("dxx(...) must be applied to a Field1D")

    # 之后可以在这里扩展其他空间项（grad/div/convection/source 等）
    raise ValueError(f"Unsupported FDM term pattern: op={expr.op}")


def _classify_term(expr: Expr, sign: float, method: Method) -> SpatialTerm:
    if method is Method.FDM:
        return _classify_term_for_fdm(expr, sign)
    # 以后 Method.FVM / FEM 在这里分发到各自的 classify_xxx
    raise ValueError(f"Method {method} not yet supported in pde_ir._classify_term")



def describe_equation(eq: Equation, method: Method) -> PDEDescription:
    """
    把 Equation(lhs, rhs) 解析成一个结构化的 PDEDescription：
      - time : 哪个场、几阶时间导
      - terms: RHS 拆出来的项列表，每项带 kind/field/系数/operator
    """
    time_desc = _parse_time_lhs(eq.lhs)

    signed_terms = _split_rhs_terms(eq.rhs)
    terms: List[SpatialTerm] = []
    for sign, term_expr in signed_terms:
        terms.append(_classify_term(term_expr, sign, method))

    return PDEDescription(
        method=method,
        time=time_desc,
        terms=terms,
    )

