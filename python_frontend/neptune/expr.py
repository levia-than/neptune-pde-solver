from .core import get_compiler

class Expr:
    def __init__(self, handle):
        self._handle = handle
    
    def _get_compiler(self):
        return get_compiler()
    
    # --- 辅助函数：把 int/float 自动转成 Expr ---
    def _as_expr(self, other):
        if isinstance(other, Expr):
            return other
        if isinstance(other, (int, float)):
            # 调用刚才在 C++ 里加的 create_constant
            new_handle = self._get_compiler().create_constant(float(other))
            return Expr(new_handle)
        raise TypeError(f"Unsupported operand type: {type(other)}")

    # --- 1. Access Op: u[0, 0] ---
    def __getitem__(self, index):
        if isinstance(index, int):
            offsets = [index]
        elif isinstance(index, (tuple, list)):
            offsets = list(index)
        else:
            raise TypeError(f"Indices must be integers or tuples, got {type(index)}")
        
        new_handle = self._get_compiler().create_access(self._handle, offsets)
        return Expr(new_handle)
    
    # --- 2. Arithmetic Ops: a + b ---
    def __add__(self, other):
        # 1. 先把 other 转成 Expr (如果是数字的话)
        other = self._as_expr(other)
        # 2. 再调用 C++
        new_handle = self._get_compiler().create_arith_add(self._handle, other._handle)
        return Expr(new_handle)

    def __sub__(self, other):
        other = self._as_expr(other)
        new_handle = self._get_compiler().create_arith_sub(self._handle, other._handle)
        return Expr(new_handle)

    def __mul__(self, other):
        other = self._as_expr(other)
        new_handle = self._get_compiler().create_arith_mul(self._handle, other._handle)
        return Expr(new_handle)

    # 支持反向运算 (比如 2.0 * u)
    def __radd__(self, other):
        return self._as_expr(other) + self
    
    def __rsub__(self, other):
        return self._as_expr(other) - self

    def __rmul__(self, other):
        return self._as_expr(other) * self
