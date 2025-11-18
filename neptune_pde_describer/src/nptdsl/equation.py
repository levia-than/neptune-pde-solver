# nptdsl/equation.py

from dataclasses import dataclass
from typing import Any, Tuple, Union

from .field import Field1D


# ======================= Expr 基础节点 =======================

@dataclass(frozen=True)
class Expr:
    """
    极简表达式节点，用于描述 PDE 里的符号运算。

    op:
      - "var"   : 变量（Field1D）
      - "const" : 常数
      - "dt"    : d_t(u)
      - "dt2"   : d_t2(u)（为将来预留）
      - "dxx"   : 二阶空间导数 d_xx(u)
      - "add"   : a + b
      - "sub"   : a - b
      - "mul"   : a * b
    """
    op: str
    args: Tuple[Any, ...]

    # ---- 运算符重载，便于构造复杂表达式 ----

    def __add__(self, other: "ExprOrScalar") -> "Expr":
        return Expr("add", (self, as_expr(other)))

    def __radd__(self, other: "ExprOrScalar") -> "Expr":
        return Expr("add", (as_expr(other), self))

    def __sub__(self, other: "ExprOrScalar") -> "Expr":
        return Expr("sub", (self, as_expr(other)))

    def __rsub__(self, other: "ExprOrScalar") -> "Expr":
        return Expr("sub", (as_expr(other), self))

    def __mul__(self, other: "ExprOrScalar") -> "Expr":
        return Expr("mul", (self, as_expr(other)))

    def __rmul__(self, other: "ExprOrScalar") -> "Expr":
        return Expr("mul", (as_expr(other), self))

    def __neg__(self) -> "Expr":
        return Expr("mul", (const(-1.0), self))

    def __repr__(self) -> str:
        return f"Expr(op={self.op!r}, args={self.args!r})"


ExprOrScalar = Union["Expr", Field1D, float, int]


def as_expr(x: ExprOrScalar) -> Expr:
    """
    把 Field1D / 标量 / Expr 统一转成 Expr。
    TimeStepper / describer 会假定 Field1D 通过 "var" 包装。
    """
    if isinstance(x, Expr):
        return x
    if isinstance(x, Field1D):
        return var(x)
    if isinstance(x, (int, float)):
        return const(float(x))
    raise TypeError(f"Cannot convert type {type(x)} to Expr")


# ======================= 基本构造函数 =======================

def var(field: Field1D) -> Expr:
    """变量节点：u -> Expr('var', (field,))."""
    return Expr("var", (field,))


def const(value: float) -> Expr:
    """常数节点：c -> Expr('const', (c,))."""
    return Expr("const", (float(value),))


def d_t(field: Field1D) -> Expr:
    """一阶时间导数：d_t(u)。"""
    return Expr("dt", (var(field),))


def d_t2(field: Field1D) -> Expr:
    """二阶时间导数：d_t2(u)，目前只是预留给解析层。"""
    return Expr("dt2", (var(field),))


def d_xx(field: Field1D) -> Expr:
    """二阶空间导数：d_xx(u)。"""
    return Expr("dxx", (var(field),))


# ======================= 方程封装 =======================

@dataclass(frozen=True)
class Equation:
    """
    PDE 的符号形式：lhs = rhs。

    例子：
        u: Field1D
        alpha = 0.1
        eq = Equation(d_t(u), alpha * d_xx(u))

    之后会交给 describer.describe_equation() 做结构化解析。
    """
    lhs: Expr
    rhs: Expr
