import neptune as np
import neptune.core as core

# 1. 定义算子 (这是我们最想测试的核心功能)
@np.linear_op_def(bounds=([0],[100]), location="cell")
def laplacian_1d(u):
    # u 是 Expr 对象
    return u[0] * 2.0 - u[-1] - u[1]

# 2. 主程序逻辑
def main_solver():
    print(">>> 开始构建 IR...")

    # --- 测试核心 DSL ---
    # 这会触发 C++ 构建 linear_opdef 的 IR
    # laplacian_1d 已经被装饰器注册为了一个 Symbol
    
    # 这会触发 assemble_matrix 指令
    matrix_H = np.assemble_matrix(laplacian_1d)
    
    print(">>> 构建完成，打印 IR:")
    print("=" * 40)
    # 打印生成的 MLIR
    print(core.get_compiler().dump())
    print("=" * 40)

if __name__ == "__main__":
    main_solver()
