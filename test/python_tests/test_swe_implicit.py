import neptune as np
import neptune.core as core
import numpy as np_numpy
import time

# ==============================================================================
# 1. 物理常数与辅助函数 (Pure Python Logic)
# 编译器会自动 Trace 并 Inline 这些逻辑
# ==============================================================================
g = 9.81

def compute_flux(h, q):
    """
    计算通量 F(U):
    Flux_h = q
    Flux_q = q^2 / h + 0.5 * g * h^2
    """
    # 为了数值稳定性，加一个小 epsilon 防止除零 (如果是干湿床)
    # h_safe = h + 1e-6 
    f_h = q
    f_q = (q * q) / h + 0.5 * g * (h * h)
    return f_h, f_q

# ==============================================================================
# 2. 线性算子定义 (用于 Preconditioner)
# 这是一个简化的线性化波动方程算子，用于加速非线性收敛
# ==============================================================================
@np.linear_op_def(bounds=([0],[100]), location="cell")
def linear_wave_op(u):
    # 这里定义一个简单的 Laplacian 或者 Wave 算子作为近似 Jacobian
    # - div(grad(u)) 
    return u[-1] - 2.0 * u[0] + u[1]

# ==============================================================================
# 3. 求解策略 (JIT Class)
# ==============================================================================
@np.jit_class
class ImplicitSWESolver:
    def __init__(self, dt, dx):
        self.dt = dt
        self.dx = dx
        
        # [炫技点 1]: 组装一个线性算子作为预条件子 (PC)
        # 虽然我们要解非线性方程，但用一个线性矩阵做 PC 能极大加速收敛
        print("[Solver] Assembling Preconditioner Matrix...")
        self.P_matrix = np.assemble_matrix(linear_wave_op)

    def step(self, h_curr, q_curr):
        """
        全隐式时间步进:
        求解非线性方程组 R(U_new) = 0
        R = U_new - U_curr + (dt/dx) * (F_right - F_left)
        """
        
        # 定义残差函数 (闭包捕获 dt, dx, h_curr, q_curr)
        # 这是一个 Local Kernel，将被编译成 C 函数指针传给 PETSc
        def swe_residual(h, q):
            # 1. 简单的中心差分通量 (或用 HLL)
            # F_{i}
            fh, fq = compute_flux(h, q)
            
            # Flux divergence (简单的中心差分)
            # div_F = (F[1] - F[-1]) / (2 * dx)
            # 注意：这里的 index 访问触发 AccessOp
            div_fh = (fh[1] - fh[-1]) / (2.0 * self.dx)
            div_fq = (fq[1] - fq[-1]) / (2.0 * self.dx)
            
            # 2. 时间导数部分
            # (U_new - U_old) / dt
            dt_h = (h[0] - h_curr[0]) / self.dt
            dt_q = (q[0] - q_curr[0]) / self.dt
            
            # 3. 总残差 = 时间项 + 空间项
            res_h = dt_h + div_fh
            res_q = dt_q + div_fq
            
            return res_h, res_q

        # [炫技点 2]: 调用非线性求解器 (JFNK + Preconditioning)
        # 编译器生成: llvm.call @neptune_nonlin_solve(...)
        # 我们传入 self.P_matrix 作为预条件矩阵
        # (注: 目前 solve_nonlinear 签名还没加 pc 参数，MVP 可以先不传，纯 JFNK)
        h_next, q_next = np.solve_nonlinear(
            swe_residual, 
            initial_guess=(h_curr, q_curr), 
            method="newton-krylov", # 指定 PETSc SNES 类型
            # pc=self.P_matrix      # 未来支持：传入预条件矩阵
        )
        
        return h_next, q_next

# ==============================================================================
# 4. 主程序驱动
# ==============================================================================
def main():
    # --- 配置 ---
    nx = 100
    dx = 1.0
    dt = 0.1 # 隐式方法允许大时间步！
    steps = 10
    
    # --- 初始化数据 (Dam Break 问题) ---
    h_data = np_numpy.ones(nx, dtype=np_numpy.float64)
    h_data[:nx//2] = 2.0 # 左边水位高
    q_data = np_numpy.zeros(nx, dtype=np_numpy.float64)
    
    print(f"Initializing SWE Problem (Nx={nx}, dt={dt})...")
    
    # --- 实例化 Solver ---
    # 这一步触发 __init__ 的 JIT (Assemble Matrix)
    solver = ImplicitSWESolver(dt=dt, dx=dx)
    
    # --- 时间循环 ---
    print("Starting Time Loop...")
    start_time = time.time()
    
    for n in range(steps):
        # 这一步触发 step 的 JIT (第一次)
        # 之后就是直接调用 C 函数
        h_data, q_data = solver.step(h_data, q_data)
        
        # 简单的进度打印
        if n % 1 == 0:
            avg_h = np_numpy.mean(h_data)
            print(f"  Step {n}: Avg Height = {avg_h:.4f}")

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.4f}s")
    
    # 简单的正确性检查
    print("Final State Sample:", h_data[45:55])

if __name__ == "__main__":
    main()
