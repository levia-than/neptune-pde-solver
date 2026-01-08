# neptune/dsl.py
from .core import get_compiler
from .expr import Expr

def apply(inputs, bounds):
    """
    @neptune.apply(inputs=[u, v], bounds=([0], [10]))
    def kernel(u, v): ...
    """
    lb, ub = bounds
    compiler = get_compiler()

    def decorator(func):
        def cpp_callback(args_handles):
            wrapped_args = [Expr(h) for h in args_handles]
            result_expr = func(*wrapped_args)
            if not isinstance(result_expr, Expr):
                raise TypeError(f"Kernel must return a Neptune Expr, got {type(result_expr)}")
            return result_expr._handle
        input_handles = [i._handle for i in inputs]
        res_handle = compiler.create_apply(input_handles, lb, ub, cpp_callback)
        
        return Expr(res_handle)
    return decorator

stencil = apply

def linear_op_def(bounds, location, name=None):
    """
    定义一个线性算子符号 (Symbol)。
    自动将用户的 scalar kernel 包装进 neptune_ir.apply 中。
    """
    compiler = get_compiler()
    
    def decorator(func):
        symbol_name = name if name else func.__name__
        def op_def_callback(op_args_handles):
            def apply_body_callback(apply_args):
                wrapped_args = [Expr(h) for h in apply_args]
                result_expr = func(*wrapped_args)
                return result_expr._handle

            lb, ub = bounds
            apply_res_handle = compiler.create_apply(
                op_args_handles, # inputs
                lb, ub,          # bounds
                apply_body_callback
            )
            
            return apply_res_handle
        compiler.create_linear_opdef(
            symbol_name, 
            bounds[0], bounds[1], 
            location, 
            op_def_callback # <-- 传入包装后的回调
        )
        
        return symbol_name 
    return decorator

# --- 3. 顶层指令 ---

def assemble_matrix(op_symbol_name):
    """
    H = neptune.assemble_matrix("laplacian")
    """
    compiler = get_compiler()
    handle = compiler.create_assemble_matrix(op_symbol_name)
    return Expr(handle) # 这里返回的是 Matrix Handle

def solve_linear(matrix, rhs, solver="cg", tol=1e-6):
    compiler = get_compiler()
    handle = compiler.create_solve_linear(matrix._handle, rhs._handle, solver, tol)
    return Expr(handle)
