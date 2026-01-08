#ifndef NEPTUNE_RUNTIME_H
#define NEPTUNE_RUNTIME_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *neptune_lin_ctx_t;
typedef void *neptune_nonlin_ctx_t;

// residual(x, f, user_ctx), x/f 都是长度 n 的连续 double 数组
typedef void (*ResidualFuncPtr)(double *x, double *f, void *user_ctx);

// ============================================================================
// MLIR LLVM lowering 使用的 memref ABI（你现在的 LLVM IR 就是这种结构）
// memref<0d, f64> : {ptr, ptr, i64}
// memref<1d, f64> : {ptr, ptr, i64, [1], [1]}
// memref<2d, f64> : {ptr, ptr, i64, [2], [2]}
// ============================================================================
typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
} NeptuneMemRef0D;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} NeptuneMemRef1D;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} NeptuneMemRef2D;

// ============================================================================
// Global init/finalize
// ============================================================================
void neptune_rt_init_noargs();
void neptune_rt_finalize();

// ============================================================================
// High-level API (你原来的接口，继续保留)
// ============================================================================

// Linear
neptune_lin_ctx_t neptune_lin_create(const char *name);
void neptune_lin_destroy(neptune_lin_ctx_t ctx);
void neptune_lin_set_options(neptune_lin_ctx_t ctx, const char *options_str);

// A: row-major dense matrix (n x n), b/x length n
void neptune_lin_solve_dense_rm(neptune_lin_ctx_t ctx, const double *A_rm,
                                const double *b, double *x, int64_t n);

// Nonlinear
neptune_nonlin_ctx_t neptune_nonlin_create(const char *name);
void neptune_nonlin_destroy(neptune_nonlin_ctx_t ctx);
void neptune_nonlin_set_options(neptune_nonlin_ctx_t ctx,
                                const char *options_str);
void neptune_nonlin_solve(neptune_nonlin_ctx_t ctx,
                          ResidualFuncPtr residual_func, double *x, int64_t n,
                          void *user_ctx);

// ============================================================================
// Low-level ABI entrypoints (给 pass 生成的 LLVM 直接调用)
// IMPORTANT: 这些函数名和签名必须和 LLVM 输出严格一致
// ============================================================================

// assemble_matrix: (keyHash, symNamePtr) -> memref<?x?xf64>
// 这里返回的 memref2d 的 allocated/aligned 会被 runtime 当作 “handle” 使用
NeptuneMemRef2D _neptune_rt_runtime_assemble_matrix(int64_t keyHash,
                                                    const char *symName,
                                                    int64_t n);

// solve_linear: (A_memref2d_expanded, b_memref1d_expanded) -> memref1d
NeptuneMemRef1D _neptune_rt_runtime_solve_linear(void *A_alloc, void *A_aligned,
                                                 int64_t A_off, int64_t A_s0,
                                                 int64_t A_s1, int64_t A_st0,
                                                 int64_t A_st1, void *b_alloc,
                                                 void *b_aligned, int64_t b_off,
                                                 int64_t b_s0, int64_t b_st0);

NeptuneMemRef1D _neptune_rt_runtime_time_advance(void *s_alloc, void *s_aligned,
                                                 int64_t s_off, int64_t s_s0,
                                                 int64_t s_st0, double dt,
                                                 int32_t method,
                                                 const char *rhs_sym);

// 可选：用于释放 MLIR lowering 返回的 malloc buffer（比如 entry / @A /
// solve_linear 的结果）
void neptune_rt_free(void *p);

NeptuneMemRef1D _neptune_rt_runtime_solve_nonlinear(
    void *x0_alloc, void *x0_aligned, int64_t x0_off, int64_t x0_s0,
    int64_t x0_st0, void *cap_alloc, void *cap_aligned, int64_t cap_off,
    int64_t cap_s0, int64_t cap_st0, const char *residual_sym, double tol,
    int64_t max_iters, const char *options_str /*可选："-snes_monitor ..."*/);

#ifdef __cplusplus
}
#endif

#endif // NEPTUNE_RUNTIME_H
