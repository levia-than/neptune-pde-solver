#ifndef NEPTUNEIR_NEPTUNEIROPS_TD
#define NEPTUNEIR_NEPTUNEIROPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "NeptuneIRDialect.td"
include "NeptuneIRAttrs.td"
include "NeptuneIRTypes.td"

//------------------------------------------------------------------------------
// neptune_ir.wrap
//------------------------------------------------------------------------------
//
// neptune_ir.wrap %buffer : memref<...> -> !neptune_ir.field<...>
//
def NeptuneIR_WrapOp
    : NeptuneIR_Op<"wrap", [Pure]> {
  let summary = "wrap a buffer (e.g., memref) into a Neptune field";
  let description = [{
    Wrap a buffer-like value (typically a memref) into a `FieldType`.
    This is the entry point from buffer-level IR into NeptuneIR.
  }];

  let arguments = (ins AnyMemRef:$buffer);
  let results   = (outs NeptuneIR_FieldType:$var_field);

  let assemblyFormat = [{
    $buffer attr-dict `:` type($buffer) `->` type($var_field)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.unwrap
//------------------------------------------------------------------------------
//
// neptune_ir.unwrap %field : !neptune_ir.field<...> -> memref<...>
//
def NeptuneIR_UnwrapOp
    : NeptuneIR_Op<"unwrap", [NoMemoryEffect]> {
  let summary = "unwrap a Neptune field back to a buffer";
  let description = [{
    Inverse of `neptune_ir.wrap`. Used when lowering NeptuneIR back to
    buffer-level dialects or interfacing with external code.
  }];

  let arguments = (ins NeptuneIR_FieldType:$var_field);
  let results   = (outs AnyMemRef:$buffer);

  let assemblyFormat = [{
    $var_field attr-dict `:` type($var_field) `->` type($buffer)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.load
//------------------------------------------------------------------------------
//
// neptune_ir.load %field : !neptune_ir.field<...> -> !neptune_ir.temp<...>
//
def NeptuneIR_LoadOp
    : NeptuneIR_Op<"load", [Pure]> {
  let summary = "load a field into a value-semantics temp";
  let description = [{
    Load a `FieldType` into a `TempType`. Bounds and location of the temp
    usually match those of the field.
  }];

  let arguments = (ins NeptuneIR_FieldType:$var_field);
  let results   = (outs NeptuneIR_TempType:$result);

  let assemblyFormat = [{
    $var_field attr-dict `:` type($var_field) `->` type($result)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.yield
//------------------------------------------------------------------------------
//
// neptune_ir.yield %val : T
//
// apply region 的 terminator.
//
def NeptuneIR_YieldOp
    : NeptuneIR_Op<"yield", [Terminator]> {
  let summary = "yield scalar result(s) from an apply region";
  let description = [{
    Terminator of a `neptune_ir.apply` region. Usually yields a single
    scalar (the updated value), but multiple results are also allowed.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];

  let extraClassDeclaration = [{
    static void build(::mlir::OpBuilder &builder,
                      ::mlir::OperationState &state) {
      build(builder, state, ::mlir::ValueRange{});
    }
  }];
}

//------------------------------------------------------------------------------
// neptune_ir.return
//------------------------------------------------------------------------------
//
// neptune_ir.return %val : T
//
// Terminator for `neptune_ir.linear_opdef` bodies.
//
def NeptuneIR_ReturnOp
    : NeptuneIR_Op<"return", [Terminator]> {
  let summary = "return results from a linear_opdef body";
  let description = [{
    Terminator for `neptune_ir.linear_opdef` regions. Mirrors
    `func.return` but stays in the NeptuneIR dialect so we can enforce
    linearity constraints before lowering to `func`.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];

  let extraClassDeclaration = [{
    static void build(::mlir::OpBuilder &builder,
                      ::mlir::OperationState &state) {
      build(builder, state, ValueRange{});
    }
  }];
}

//------------------------------------------------------------------------------
// neptune_ir.apply
//------------------------------------------------------------------------------
//
// neptune_ir.apply (%arg0, %arg1, ...)
//     { bounds = #neptune_ir.bounds<...>,
//       shape? = #neptune_ir.stencil_shape<...> }
//   : (!neptune_ir.temp<...>, ...) -> !neptune_ir.temp<...>
//   {
//     ^bb0(%i0 : index, %i1 : index, ...):
//       %u = neptune_ir.access %arg0[0] : ... -> f64
//       ...
//       neptune_ir.yield %u_new : f64
//   }
//
// 注意：SingleBlockImplicitTerminator = NeptuneIR_YieldOp
//
def NeptuneIR_ApplyOp
    : NeptuneIR_Op<"apply",
        [Pure, SingleBlock, IsolatedFromAbove,  
        SingleBlockImplicitTerminator<"mlir::Neptune::NeptuneIR::YieldOp">]> {
  let summary = "structured stencil kernel over a domain";
  let description = [{
    `neptune_ir.apply` expresses a stencil kernel over a logical domain
    (given by `bounds`). Inputs are Temp fields; the result is a Temp
    field. Inside the region you only operate on scalars and use
    `neptune_ir.access` to read neighbors.

    Optionally, `shape` can encode the stencil pattern as a
    `StencilShapeAttr`, which is convenient for optimization passes.
  }];

  // operands
  let arguments = (ins
    Variadic<NeptuneIR_TempType>:$inputs,
    NeptuneIR_BoundsAttr:$bounds,
    OptionalAttr<NeptuneIR_StencilShapeAttr>:$shape
  );

  let results = (outs NeptuneIR_TempType:$result);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict-with-keyword
      `:` functional-type($inputs, $result)
      $body
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.access
//------------------------------------------------------------------------------
//
// neptune_ir.access %tmp[<offsets>] : !neptune_ir.temp<...> -> elementType
//
def NeptuneIR_AccessOp
    : NeptuneIR_Op<"access", [Pure]> {
  let summary = "access a neighbor element from a temp at given offsets";
  let description = [{
    Access an element of a `TempType` at a compile-time constant offset
    relative to the current logical point. Offsets are given as a
    DenseI64ArrayAttr like [0], [-1], [1] in 1D, or [0,1,0] in 3D.
  }];

  let arguments = (ins
    NeptuneIR_TempType:$input,
    DenseI64ArrayAttr:$offsets
  );

  // 实际上 result 类型应该等于 input.elementType，C++ verifier 里可以检查。
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $input $offsets attr-dict
      `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.store
//------------------------------------------------------------------------------
//
// neptune_ir.store %tmp to %field
//   { bounds? = #neptune_ir.bounds<...> }
//   : !neptune_ir.temp<...> to !neptune_ir.field<...>
//
def NeptuneIR_StoreOp
    : NeptuneIR_Op<"store",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "store a temp field into a destination field";
  let description = [{
    Store a `TempType` into a `FieldType`. Optional `bounds` restricts
    the write to a sub-domain (e.g., interior or boundary region).
  }];

  let arguments = (ins
    NeptuneIR_TempType:$value,
    NeptuneIR_FieldType:$var_field,
    OptionalAttr<NeptuneIR_BoundsAttr>:$bounds
  );

  let assemblyFormat = [{
    $value `to` $var_field attr-dict
      `:` type($value) `to` type($var_field)
  }];

  let hasVerifier = 1;
}


//------------------------------------------------------------------------------
// neptune_ir.reduce
//------------------------------------------------------------------------------
//
// neptune_ir.reduce %tmp in {bounds?} {kind = "sum"}
//   : !neptune_ir.temp<...> -> f64
//
// 场级别 reduction，用于 dot/norm 等线性代数骨架。
// kind 约定： "sum" | "max" | "min" | "l1" | "l2" 等。
//
def NeptuneIR_ReduceOp
    : NeptuneIR_Op<"reduce", [Pure]> {
  let summary = "reduce a temp field over an optional sub-domain";
  let description = [{
    Reduce a `TempType` over the given domain to a scalar. This is the
    building block for dot products, norms, and other PDE/solver-level
    reductions.

    The `kind` attribute selects the reduction operator, e.g. "sum",
    "max", "min", "l1", "l2". Exact semantics are enforced by the
    C++ verifier / implementation.
  }];

  let arguments = (ins
    NeptuneIR_TempType:$input,
    OptionalAttr<NeptuneIR_BoundsAttr>:$bounds,
    StrAttr:$kind
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $input (`in` $bounds^)? attr-dict
      `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.linear_opdef
//------------------------------------------------------------------------------
//
// 线性算子定义（symbol），类似 func.func：
//   neptune_ir.linear_opdef @A : (!neptune_ir.temp<...>) -> !neptune_ir.temp<...> {
//     ^bb0(%in : !neptune_ir.temp<...>):
//       ...  // 典型做法是内部用 neptune_ir.apply 搞 stencil
//       neptune_ir.yield %out : !neptune_ir.temp<...>
//   }
//
// 之后用 neptune_ir.apply_linear @A(%x) 调用。
//
// 要求：
//  - 无 SSA 结果（符号）
//  - 有 'sym_name' StringAttr（SymbolNameAttr）
//
def NeptuneIR_LinearOpDefOp
    : NeptuneIR_Op<"linear_opdef",
        [IsolatedFromAbove,
         Symbol,
         FunctionOpInterface,
         SingleBlockImplicitTerminator<"mlir::Neptune::NeptuneIR::ReturnOp">]> {
  let summary = "define a linear operator over temp fields";
  let description = [{
    Define a named linear operator over Temp fields, similar to a
    function. The signature is carried in a `function_type` attribute
    that must be a function type `(Temp...) -> (Temp...)`.

    The body is a region whose entry block arguments match the
    function type inputs. Implementations typically use
    `neptune_ir.apply` + `neptune_ir.access` inside to describe the
    action of the operator, and `neptune_ir.returnOp` to return the
    output Temp(s).

    This op is a Symbol and is referenced by `neptune_ir.apply_linear`
    and `neptune_ir.assemble_matrix`.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$function_type        // must be a FunctionType
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $sym_name `:` $function_type attr-dict-with-keyword $body
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    using Op::getOperation;
    ::mlir::FunctionType getSignature() {
      return ::llvm::cast<::mlir::FunctionType>(getFunctionType());
    }
    ::mlir::Region *getCallableRegion() { return &getBody(); }
    ::llvm::ArrayRef<::mlir::Type> getCallableResults() {
      return getSignature().getResults();
    }
    ::llvm::ArrayRef<::mlir::Type> getCallableArgumentTypes() {
      return getSignature().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getSignature().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getSignature().getResults();
    }
    ::mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(::mlir::ArrayAttr) {}
    ::mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setResAttrsAttr(::mlir::ArrayAttr) {}
    ::mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    ::mlir::Attribute removeResAttrsAttr() { return nullptr; }
    ::mlir::Type cloneTypeWith(::mlir::TypeRange inputs,
                               ::mlir::TypeRange results) {
      return ::mlir::FunctionType::get(getContext(), inputs, results);
    }
  }];
}


def NeptuneIR_NonlinearOpDefOp
    : NeptuneIR_Op<"nonlinear_opdef",
        [IsolatedFromAbove,
         Symbol,
         FunctionOpInterface,
         SingleBlockImplicitTerminator<"mlir::Neptune::NeptuneIR::ReturnOp">]> {
  let summary = "define a non linear operator over temp fields";
  let description = [{
    Define a named non linear operator over Temp fields, similar to a
    function. The signature is carried in a `function_type` attribute
    that must be a function type `(Temp...) -> (Temp...)`.

    The body is a region whose entry block arguments match the
    function type inputs. Implementations typically use
    `neptune_ir.apply` + `neptune_ir.access` inside to describe the
    action of the operator, and `neptune_ir.return` to return the
    output Temp(s).

    This op is a Symbol and is referenced by `neptune_ir.solve_nonlinear`
    and `neptune_ir.apply_nonlinear`.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$function_type        // must be a FunctionType
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $sym_name `:` $function_type attr-dict-with-keyword $body
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    using Op::getOperation;
    ::mlir::FunctionType getSignature() {
      return ::llvm::cast<::mlir::FunctionType>(getFunctionType());
    }
    ::mlir::Region *getCallableRegion() { return &getBody(); }
    ::llvm::ArrayRef<::mlir::Type> getCallableResults() {
      return getSignature().getResults();
    }
    ::llvm::ArrayRef<::mlir::Type> getCallableArgumentTypes() {
      return getSignature().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getSignature().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getSignature().getResults();
    }
    ::mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(::mlir::ArrayAttr) {}
    ::mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setResAttrsAttr(::mlir::ArrayAttr) {}
    ::mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    ::mlir::Attribute removeResAttrsAttr() { return nullptr; }
    ::mlir::Type cloneTypeWith(::mlir::TypeRange inputs,
                               ::mlir::TypeRange results) {
      return ::mlir::FunctionType::get(getContext(), inputs, results);
    }
  }];
}

//------------------------------------------------------------------------------
//  NEW: neptune_ir.apply_linear
//------------------------------------------------------------------------------
//
// neptune_ir.apply_linear @A(%in)
//   : (!neptune_ir.temp<...>) -> !neptune_ir.temp<...>
//
// 把线性算子当成 “matrix-free SpMV” 来用： y = A(x)。
// A 由 `neptune_ir.linear_opdef` 定义。
//
def NeptuneIR_ApplyLinearOp
    : NeptuneIR_Op<"apply_linear", [Pure]> {
  let summary = "matrix-free application of a linear operator";
  let description = [{
    Apply a previously defined linear operator (from
    `neptune_ir.linear_opdef`) in a matrix-free fashion. This is the
    core primitive used by iterative solvers (CG/GMRES/etc.) in
    matrix-free mode.

    The `op` attribute is a `SymbolRefAttr` pointing to a
    `neptune_ir.linear_opdef`.
  }];

  let arguments = (ins
    SymbolRefAttr:$op,
    Variadic<NeptuneIR_TempType>:$inputs,
    OptionalAttr<NeptuneIR_BoundsAttr>:$bounds
  );

  let results = (outs Variadic<NeptuneIR_TempType>:$results);

  let assemblyFormat = [{
    $op `(` $inputs `)` attr-dict
      `:` functional-type($inputs, $results)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
//  NEW: neptune_ir.apply_linear
//------------------------------------------------------------------------------
//
// neptune_ir.apply_linear @A(%in)
//   : (!neptune_ir.temp<...>) -> !neptune_ir.temp<...>
//
// 把线性算子当成 “matrix-free SpMV” 来用： y = A(x)。
// A 由 `neptune_ir.linear_opdef` 定义。
//
def NeptuneIR_ApplyNonLinearOp
    : NeptuneIR_Op<"apply_nonlinear", [Pure]> {
  let summary = "matrix-free application of a non linear operator";
  let description = [{
    Apply a previously defined non linear operator (from
    `neptune_ir.linear_opdef`) in a matrix-free fashion. This is the
    core primitive used by iterative solvers (CG/GMRES/etc.) in
    matrix-free mode.

    The `op` attribute is a `SymbolRefAttr` pointing to a
    `neptune_ir.linear_opdef`.
  }];

  let arguments = (ins
    SymbolRefAttr:$op,
    Variadic<NeptuneIR_TempType>:$inputs,
    OptionalAttr<NeptuneIR_BoundsAttr>:$bounds
  );

  let results = (outs Variadic<NeptuneIR_TempType>:$results);

  let assemblyFormat = [{
    $op `(` $inputs `)` attr-dict
      `:` functional-type($inputs, $results)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.as_tensor
//------------------------------------------------------------------------------
//
// neptune_ir.as_tensor %tmp
//   : !neptune_ir.temp<...> -> tensor<...>
//
// 把 Neptune 的 Temp 映射到 tensor，方便用 linalg / stablehlo 等。
// 具体 shape 由 TempType + bounds 推导，C++ verifier 保证。
//
def NeptuneIR_AsTensorOp
    : NeptuneIR_Op<"as_tensor", [Pure]> {
  let summary = "view a Neptune temp as a tensor";
  let description = [{
    View a `TempType` as a tensor value, typically to hand it off to
    `linalg` / `tensor` / `stablehlo`-based linear algebra and NN
    code. The result type must be a ranked tensor whose element type
    matches that of the Temp.

    The exact shape mapping is determined by the Temp's bounds and
    is enforced in the C++ verifier.
  }];

  let arguments = (ins NeptuneIR_TempType:$input);

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input attr-dict
      `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.from_tensor
//------------------------------------------------------------------------------
//
// neptune_ir.from_tensor %t
//   : tensor<...> -> !neptune_ir.temp<...>
//
// 逆操作：把 tensor 转回 Temp。
// 通常在 linarg / matmul 之后回到 Neptune 世界。
//
def NeptuneIR_FromTensorOp
    : NeptuneIR_Op<"from_tensor", [Pure]> {
  let summary = "wrap a tensor back into a Neptune temp";
  let description = [{
    Inverse of `neptune_ir.as_tensor`. Wrap a tensor value back into
    a `TempType`, e.g., after a matmul or other linalg computation.

    The Temp's element type must match the tensor element type; bounds
    / logical domain are carried via the TempType and validated in the
    C++ verifier.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs NeptuneIR_TempType:$result);

  let assemblyFormat = [{
    $input attr-dict
      `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.assemble_matrix
//------------------------------------------------------------------------------
//
// neptune_ir.assemble_matrix @A
//   : memref<?x?xf64>
//
// 从线性算子定义构造（装配）一个矩阵表示，交给后续 pipeline。
//
def NeptuneIR_AssembleMatrixOp
    : NeptuneIR_Op<"assemble_matrix",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "assemble a matrix from a linear operator";
  let description = [{
    Assemble a matrix representation of a `neptune_ir.linear_opdef`.
    MVP restriction: result must be `memref<?x?xf64>`.

    The C++ verifier ensures that the `op` symbol refers to a
    compatible `linear_opdef`.
  }];

  let arguments = (ins
    SymbolRefAttr:$op,
    OptionalAttr<NeptuneIR_BoundsAttr>:$bounds
  );

  let results = (outs AnyMemRef:$matrix);

  let assemblyFormat = [{
    $op attr-dict
      `:` type($matrix)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.solve_linear
//------------------------------------------------------------------------------
//
// neptune_ir.solve_linear %A, %b
//   { solver = "cg", tol = 1.0e-8, max_iters = 1000 }
//   : (memref<?x?xf64>, !neptune_ir.temp<...>)
//   -> !neptune_ir.temp<...>
//
// 线性方程组求解：A x = b。
// A 必须是 assemble_matrix 生成的 memref<?x?xf64>（MVP assembled-only）。
//
def NeptuneIR_SolveLinearOp
    : NeptuneIR_Op<"solve_linear",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "solve a linear system A x = b";
  let description = [{
    Solve a linear system `A x = b` where `A` is a dense-like memref
    assembled via `neptune_ir.assemble_matrix`, and `b` is a Temp
    field representing the right-hand side.

    Attributes:
      - `solver` (string): name of the solver, e.g., "cg", "gmres",
        "bicgstab".
      - `tol` (float): convergence tolerance.
      - `max_iters` (integer): maximum number of iterations.

    The C++ implementation may call into external libraries (MKL,
    PETSc, etc.) or use Neptune-specific iterative solvers. This op
    is modeled as having memory effects because it may allocate
    temporaries or interact with external solver state.
  }];

  let arguments = (ins
    AnyMemRef:$system,
    NeptuneIR_TempType:$rhs,
    OptionalAttr<StrAttr>:$solver,
    OptionalAttr<F64Attr>:$tol,
    OptionalAttr<I64Attr>:$max_iters
  );

  let results = (outs NeptuneIR_TempType:$result);

  let assemblyFormat = [{
    $system `,` $rhs attr-dict
      `:` type($system) `,` type($rhs)
      `->` type($result)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
//  neptune_ir.solve_nonlinear
//------------------------------------------------------------------------------
//
// neptune_ir.solve_nonlinear @F(%u0)
//   { method = "newton-krylov", tol = 1.0e-8, max_iters = 50 }
//   : !neptune_ir.temp<...> -> !neptune_ir.temp<...>
//
// 非线性残量 F(u) = 0 的求解入口，适合隐式 MOL 等场景。
// F 一般是一个 symbol（可以是另一种 opdef），内部调用线性 solver。
//
def NeptuneIR_SolveNonlinearOp
    : NeptuneIR_Op<"solve_nonlinear",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "solve a nonlinear system F(u) = 0";
  let description = [{
    Solve a nonlinear system `F(u) = 0` starting from an initial
    guess `u0`. The `residual` symbol refers to an operation that
    computes the residual Temp field `F(u)` given a Temp field `u`.
    Optionally, a `jacobian` symbol may be provided to supply a
    linearization.

    Attributes:
      - `method` (string): e.g., "newton", "newton-krylov".
      - `tol` (float): nonlinear convergence tolerance.
      - `max_iters` (integer): maximum number of nonlinear iterations.

    This op provides the high-level entry point for implicit
    Method-of-Lines (MOL) style solves and other nonlinear PDE
    formulations.
  }];

  let arguments = (ins
    SymbolRefAttr:$residual,
    OptionalAttr<SymbolRefAttr>:$jacobian,
    NeptuneIR_TempType:$initial,
    Variadic<AnyType>:$captures,
    OptionalAttr<StrAttr>:$method,
    OptionalAttr<F64Attr>:$tol,
    OptionalAttr<I64Attr>:$max_iters
  );

  let results = (outs NeptuneIR_TempType:$result);

  let assemblyFormat = [{
    $residual `(` $initial (`,` $captures^)? `)` attr-dict
      `:` type($initial) (`,` type($captures)^)? `->` type($result)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
//  neptune_ir.time_advance
//------------------------------------------------------------------------------
//
// neptune_ir.time_advance %state, %dt : (!neptune_ir.temp<...>, f64) -> !neptune_ir.temp<...>
//
// 标记一次时间推进，dt 由外部驱动提供；具体时间积分细节可在 region/后续 pass 中处理。
//
def NeptuneIR_TimeAdvanceOp
    : NeptuneIR_Op<"time_advance",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let arguments = (ins
    NeptuneIR_TempType:$state,
    F64:$dt,
    NeptuneIR_TimeMethodAttr:$method,

    OptionalAttr<SymbolRefAttr>:$system,
    OptionalAttr<SymbolRefAttr>:$rhs,
    OptionalAttr<SymbolRefAttr>:$residual,
    OptionalAttr<SymbolRefAttr>:$jacobian,

    OptionalAttr<StrAttr>:$solver,
    OptionalAttr<F64Attr>:$tol,
    OptionalAttr<I64Attr>:$max_iters
  );

  let results = (outs NeptuneIR_TempType:$result);

  let assemblyFormat = [{
    $state `,` $dt attr-dict
      `:` type($state) `,` type($dt)
      `->` type($result)
  }];

  let hasVerifier = 1;
}


//------------------------------------------------------------------------------
//  neptune_ir.time_advance_runtime
//------------------------------------------------------------------------------
//
// neptune_ir.time_advance_runtime %state, %dt : (!neptune_ir.temp<...>, f64) -> !neptune_ir.temp<...>
//
def NeptuneIR_TimeAdvanceRuntimeOp
    : NeptuneIR_Op<"time_advance_runtime",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let arguments = (ins
    NeptuneIR_TempType:$state,
    F64:$dt,
    NeptuneIR_TimeMethodAttr:$method,

    OptionalAttr<SymbolRefAttr>:$system,
    OptionalAttr<SymbolRefAttr>:$rhs,
    OptionalAttr<SymbolRefAttr>:$residual,
    OptionalAttr<SymbolRefAttr>:$jacobian,

    OptionalAttr<StrAttr>:$solver,
    OptionalAttr<F64Attr>:$tol,
    OptionalAttr<I64Attr>:$max_iters
  );

  let results = (outs NeptuneIR_TempType:$result);

  let assemblyFormat = [{
    $state `,` $dt attr-dict
      `:` type($state) `,` type($dt)
      `->` type($result)
  }];

  let hasVerifier = 1;
}

#endif
