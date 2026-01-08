import neptune as np
import numpy as np_numpy

# 1. Physics
@np.linear_op_def(bounds=([0],[100]), location="cell")
def laplacian(u):
    return u[0]*2 - u[-1] - u[1]

# 2. JIT Class
@np.jit_class
class HeatSolver:
    def __init__(self, dt):
        self.dt = dt
        # 这一步在 JIT 编译时会生成 neptune_ir.assemble_matrix 指令
        # 并把结果存入 Global 变量 (或者暂时作为常量)
        self.H = np.assemble_matrix(laplacian)

    def step(self, u):
        # 这里的 u 是编译器给的占位符
        # 逻辑会被转换成 neptune_ir.solve_linear
        return np.solve_linear(self.H, u, solver="cg")

# 3. Driver
def main():
    print(">>> Initializing Solver...")
    solver = HeatSolver(dt=0.01) # 此时什么都没发生
    
    print(">>> Preparing Data...")
    # 模拟真实数据 (不再需要 dummy!)
    # 我们需要造一个 Expr 包装器来欺骗现有的 createAccess 检查
    # 等后续有了 Data Marshaling，这里就是纯 NumPy
    compiler = np.core.get_compiler()
    real_data_proxy = np.Expr(compiler.create_wrap(None, "float64"))
    
    print(">>> Calling step() (Triggers JIT)...")
    result = solver.step(real_data_proxy)
    
    print(">>> Result:", result)
    
    # 打印 IR 验证
    print("\n[Generated IR]")
    print(np.core.get_compiler().dump())

if __name__ == "__main__":
    main()
