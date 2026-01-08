#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"
#include "Utils/FuncNameHelper.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#include "llvm/Support/CommandLine.h"

namespace {
enum class LinearLoweringMode { Assembled, MatrixFree };

static llvm::cl::opt<LinearLoweringMode> clLinearMode(
    "neptune-rt-linear-mode",
    llvm::cl::desc("Neptune runtime lowering strategy for linear solve"),
    llvm::cl::values(
        clEnumValN(
            LinearLoweringMode::Assembled, "assembled",
            "Lower assemble_matrix + solve_linear to assembled runtime calls"),
        clEnumValN(
            LinearLoweringMode::MatrixFree, "matrix-free",
            "Lower to matrix-free runtime (runtime not implemented yet)")),
    llvm::cl::init(LinearLoweringMode::Assembled));

} // namespace

namespace {
using namespace mlir::Neptune::NeptuneIR;
namespace func = mlir::func;

static func::FuncOp getOrCreateRuntimeFunc(Operation *anchor, StringRef name,
                                           ArrayRef<Type> inputs,
                                           ArrayRef<Type> results,
                                           PatternRewriter &rewriter) {
  ModuleOp m = anchor->getParentOfType<ModuleOp>();
  if (auto f = m.lookupSymbol<func::FuncOp>(name))
    return f;

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(m.getBody());
  auto fn = rewriter.create<func::FuncOp>(
      m.getLoc(), name, FunctionType::get(m.getContext(), inputs, results));
  fn.setPrivate();
  return fn;
}

// 把 "abc" 变成 "abc\0"，并放到 llvm.global 里，返回 ptr(指向该全局的地址)
static mlir::Value getOrCreateCStringPtr(mlir::Operation *anchor,
                                         mlir::PatternRewriter &rewriter,
                                         mlir::Location loc,
                                         llvm::StringRef globalName,
                                         llvm::StringRef strNoNul) {
  auto module = anchor->getParentOfType<mlir::ModuleOp>();
  auto *ctx = rewriter.getContext();

  auto i8Ty = mlir::IntegerType::get(ctx, 8);
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx); // LLVM22: opaque ptr

  std::string bytes = strNoNul.str();
  bytes.push_back('\0'); // 必须 NUL terminated

  auto arrTy = mlir::LLVM::LLVMArrayType::get(i8Ty, bytes.size());

  // 复用同名 global（避免重复创建）
  if (!module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::GlobalOp>(
        loc, arrTy,
        /*isConstant=*/true, mlir::LLVM::Linkage::Internal, globalName,
        rewriter.getStringAttr(bytes));
  }

  // address_of @global : !llvm.ptr   (opaque ptr 下可直接当 i8* 用)
  return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ptrTy, globalName);
}

// 可选：空字符串全局（用于 optsPtr，避免传 null）
static mlir::Value getEmptyCStringPtr(mlir::Operation *anchor,
                                      mlir::PatternRewriter &rewriter,
                                      mlir::Location loc) {
  return getOrCreateCStringPtr(anchor, rewriter, loc, "__neptune_empty_cstr",
                               "" /*-> "\0"*/);
}

// 把 dt 转成 f64：
// - f64: 原样
// - f16/f32/bf16: extf 到 f64
// - index: index_cast 到 i64，再 sitofp 到 f64
// - iN: (ext/trunc) 到 i64，再 sitofp 到 f64
static mlir::Value castDtToF64(mlir::Value dt, mlir::PatternRewriter &rewriter,
                               mlir::Location loc) {
  auto f64Ty = rewriter.getF64Type();

  mlir::Type t = dt.getType();
  if (t == f64Ty)
    return dt;

  if (auto ft = dyn_cast<mlir::FloatType>(t)) {
    return rewriter.create<mlir::arith::ExtFOp>(loc, f64Ty, dt);
  }

  if (isa<mlir::IndexType>(t)) {
    auto i64Ty = rewriter.getI64Type();
    auto asI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Ty, dt);
    return rewriter.create<mlir::arith::SIToFPOp>(loc, f64Ty, asI64);
  }

  if (auto it = dyn_cast<mlir::IntegerType>(t)) {
    auto i64Ty = rewriter.getI64Type();
    mlir::Value asI64 = dt;
    if (it.getWidth() < 64) {
      // 默认按 signed 处理（如果你想支持 unsigned，就需要额外 attr 标记）
      asI64 = rewriter.create<mlir::arith::ExtSIOp>(loc, i64Ty, dt);
    } else if (it.getWidth() > 64) {
      // verifier 已经挡住了，这里只是兜底
      asI64 = rewriter.create<mlir::arith::TruncIOp>(loc, i64Ty, dt);
    }
    return rewriter.create<mlir::arith::SIToFPOp>(loc, f64Ty, asI64);
  }

  // 理论上 verifier 已经保证不会到这
  return {};
}

struct AssembleMatrixLowering : OpRewritePattern<AssembleMatrixOp> {
  AssembleMatrixLowering(MLIRContext *ctx, StringRef prefix)
      : OpRewritePattern(ctx), prefix(prefix.str()) {}
  std::string prefix;

  LogicalResult matchAndRewrite(AssembleMatrixOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp m = op->getParentOfType<ModuleOp>();
    auto i64Ty = rewriter.getI64Type();

    // 1) hash
    int64_t keyHash = 0;
    if (auto symRef = dyn_cast<SymbolRefAttr>(op.getOpAttr())) {
      if (auto def = SymbolTable::lookupNearestSymbolFrom(
              op, symRef.getRootReference())) {
        if (auto h = def->getAttrOfType<IntegerAttr>("structure_key_hash"))
          keyHash = h.getInt();
      }
    }
    Value key = rewriter.create<arith::ConstantIntOp>(op.getLoc(), keyHash, 64);

    // 2) op symbol string
    auto symRef = dyn_cast<SymbolRefAttr>(op.getOpAttr());
    if (!symRef)
      return op.emitOpError("requires SymbolRefAttr 'op'");
    std::string opName = symRef.getRootReference().str(); // "A"

    // generate a unique global symbol name for the string (avoid collisions)
    std::string gsym = "_neptune_sym_" + opName;
    Value opNamePtr =
        getGlobalStringPtr(op.getLoc(), m, rewriter, gsym, opName);

    // 3) runtime callee
    // 注意：这里函数参数类型要包含 i8*（LLVM ptr）
    auto i8PtrTy = opNamePtr.getType();
    auto callee = getOrCreateRuntimeFunc(op, prefix + "assemble_matrix",
                                         {i64Ty, i8PtrTy, i64Ty},
                                         {op.getMatrix().getType()}, rewriter);

    auto matrixSize = op.getBounds()->getUb()[0] - op.getBounds()->getLb()[0];
    auto matrixSizeValue = rewriter.create<arith::ConstantIntOp>(
        op->getLoc(), matrixSize, rewriter.getI64Type().getWidth());
    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), callee.getSymName(), TypeRange{op.getMatrix().getType()},
        ValueRange{key, opNamePtr, matrixSizeValue});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct SolveLinearLowering : OpRewritePattern<SolveLinearOp> {
  SolveLinearLowering(MLIRContext *ctx, StringRef prefix)
      : OpRewritePattern(ctx), prefix(prefix.str()) {}
  std::string prefix;

  LogicalResult matchAndRewrite(SolveLinearOp op,
                                PatternRewriter &rewriter) const override {
    std::string nameStr = prefix + "solve_linear";
    StringRef name = nameStr;

    auto systemTy = op.getSystem().getType();
    auto rhsTy = op.getRhs().getType();
    auto resTy = op.getResult().getType();

    auto callee =
        getOrCreateRuntimeFunc(op, name, {systemTy, rhsTy}, {resTy}, rewriter);

    auto call = rewriter.create<func::CallOp>(
        op.getLoc(), callee.getSymName(), TypeRange{resTy},
        ValueRange{op.getSystem(), op.getRhs()});

    // 把 solver/tol/max_iters 保留成 metadata（runtime 也可忽略）
    for (auto n : {"solver", "tol", "max_iters"}) {
      if (Attribute a = op->getAttr(n))
        call->setAttr(n, a);
    }

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

static std::string makeSolveNonlinearSym(StringRef prefix, int rank, int caps) {
  // 你喜欢 0d/1d/2d 也行
  return (prefix + "solve_nonlinear_" + std::to_string(rank) + "d_" +
          std::to_string(caps) + "cap")
      .str();
}

static Value makeNullPtr(Location loc, PatternRewriter &rewriter) {
  auto ptrTy =
      mlir::LLVM::LLVMPointerType::get(rewriter.getContext()); // opaque ptr
  return rewriter.create<mlir::LLVM::ZeroOp>(loc, ptrTy);      // null pointer
}

// -------------- helpers --------------
static std::optional<int> getNeptuneRank(mlir::Type ty) {
  // 如果已经被你后面的 pass 变成 memref 了，也兼容
  if (auto mem = mlir::dyn_cast<mlir::MemRefType>(ty))
    return (int)mem.getRank();

  // 你现在真实的场景：TempType，rank 来自 bounds 维度
  if (auto tmp = mlir::dyn_cast<mlir::Neptune::NeptuneIR::TempType>(ty)) {
    auto b = tmp.getBounds();
    if (!b)
      return std::nullopt;
    return (int)b.getLb().asArrayRef().size(); // 0D => size()==0
  }

  return std::nullopt;
}

static std::optional<mlir::Neptune::NeptuneIR::BoundsAttr>
getNeptuneBounds(mlir::Type ty) {
  if (auto tmp = mlir::dyn_cast<mlir::Neptune::NeptuneIR::TempType>(ty))
    return tmp.getBounds();
  return std::nullopt;
}

struct SolveNonlinearLowering : OpRewritePattern<SolveNonlinearOp> {
  SolveNonlinearLowering(MLIRContext *ctx, StringRef prefix)
      : OpRewritePattern(ctx), prefix(prefix.str()) {}
  std::string prefix;

  LogicalResult matchAndRewrite(SolveNonlinearOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // ---------- 1) rank / caps ----------
    auto initTy = op.getInitial().getType();

    auto rankOpt = getNeptuneRank(initTy);
    if (!rankOpt)
      return rewriter.notifyMatchFailure(
          op, "solve_nonlinear: initial must be !neptune_ir.temp (with bounds) "
              "or memref");

    int rank = *rankOpt; // 0/1/2/...
    int caps =
        (int)op.getCaptures().size(); // 0/1/2 (你 runtime 只做 0~2 就在这卡死)

    if (caps < 0 || caps > 2)
      return rewriter.notifyMatchFailure(
          op, "solve_nonlinear: only supports 0..2 captures (MVP)");

    // ---------- 2) 强校验：captures 必须是 temp/memref 且 rank 一致 ----------
    for (auto [i, v] : llvm::enumerate(op.getCaptures())) {
      auto r = getNeptuneRank(v.getType());
      if (!r)
        return rewriter.notifyMatchFailure(
            op,
            "solve_nonlinear: capture must be temp (with bounds) or memref");
      if (*r != rank)
        return rewriter.notifyMatchFailure(
            op, "solve_nonlinear: capture rank must match initial rank");
    }

    // （可选但强烈建议）如果 initial 是 TempType，也校验 bounds 的维度一致性
    if (auto bOpt = getNeptuneBounds(initTy)) {
      auto lbs = bOpt->getLb().asArrayRef();
      auto ubs = bOpt->getUb().asArrayRef();
      if ((int)lbs.size() != rank || (int)ubs.size() != rank)
        return rewriter.notifyMatchFailure(op,
                                           "solve_nonlinear: malformed bounds");
    }

    std::string sym = makeSolveNonlinearSym(prefix, rank, caps);

    // ---------- 2) 构造 runtime 函数签名 ----------
    Type outTy = op.getResult().getType();

    SmallVector<Type> argTys;
    argTys.push_back(op.getInitial().getType());
    for (Value c : op.getCaptures())
      argTys.push_back(c.getType());

    // 额外 runtime 参数（你 runtime 需要啥就塞啥，顺序必须一致）
    // residual_sym: i8*
    // tol: f64
    // max_iters: i64
    // petsc_options: i8*
    auto *ctx = rewriter.getContext();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

    argTys.push_back(ptrTy);
    argTys.push_back(rewriter.getF64Type());
    argTys.push_back(rewriter.getI64Type());
    argTys.push_back(ptrTy);

    auto callee = getOrCreateRuntimeFunc(op, sym, argTys, {outTy}, rewriter);

    // ---------- 3) 组装 operands ----------
    SmallVector<Value> args;
    args.push_back(op.getInitial());
    for (Value c : op.getCaptures())
      args.push_back(c);

    // residual_sym / options：这里你后面要换成真正的 global string 指针
    // residual_sym: 从 op 的 SymbolRefAttr materialize 成 const char*
    auto residualAttr = op.getResidualAttr(); // SymbolRefAttr
    auto residualName =
        residualAttr.getRootReference().getValue(); // "ac_residual"

    // 生成一个稳定的 global 名称（避免奇怪字符）
    std::string gname =
        (llvm::Twine("__neptune_residual_") + residualName).str();
    for (char &c : gname) {
      if (!(isalnum((unsigned char)c) || c == '_'))
        c = '_';
    }

    mlir::Value residualPtr = getOrCreateCStringPtr(op.getOperation(), rewriter,
                                                    loc, gname, residualName);

    // petsc_options：如果你还没设计这个 attr，就先传 "" 而不是 null（更稳）
    mlir::Value optsPtr = getEmptyCStringPtr(op.getOperation(), rewriter, loc);

    // 如果你未来加了 attr，比如 op->getAttrOfType<StringAttr>("petsc_options")
    // 也可以这样：
    // if (auto s = op->getAttrOfType<mlir::StringAttr>("petsc_options")) {
    //   optsPtr = getOrCreateCStringPtr(op.getOperation(), rewriter, loc,
    //                                   "__neptune_opts_" + mangle(s),
    //                                   s.getValue());
    // }

    double tol = op.getTolAttr() ? op.getTolAttr().getValueAsDouble() : 0.0;
    int64_t maxIt = op.getMaxItersAttr() ? op.getMaxItersAttr().getInt() : 0;

    Value tolV = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(tol));
    Value maxItV = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(maxIt));

    args.push_back(residualPtr);
    args.push_back(tolV);
    args.push_back(maxItV);
    args.push_back(optsPtr);

    auto call = rewriter.create<func::CallOp>(loc, callee.getSymName(),
                                              TypeRange{outTy}, args);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct TimeAdvanceRuntimeLowering
    : mlir::OpRewritePattern<TimeAdvanceRuntimeOp> {
  TimeAdvanceRuntimeLowering(mlir::MLIRContext *ctx, mlir::StringRef prefix)
      : mlir::OpRewritePattern<TimeAdvanceRuntimeOp>(ctx),
        prefix(prefix.str()) {}
  std::string prefix;

  mlir::LogicalResult
  matchAndRewrite(TimeAdvanceRuntimeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<mlir::ModuleOp>();

    // 原 op types
    auto stTy = op.getState().getType();
    auto outTy = op.getResult().getType();

    // 统一 dt ABI：f64
    mlir::Value dtF64 = castDtToF64(op.getDt(), rewriter, loc);
    if (!dtF64)
      return rewriter.notifyMatchFailure(op, "failed to cast dt to f64");

    // method: 默认 marker=0
    int32_t method = 0;
    if (auto a = op->getAttrOfType<mlir::IntegerAttr>("method"))
      method = (int32_t)a.getInt();
    // 如果你之后用 EnumAttr，这里也能兼容：它本质也是 IntegerAttr

    mlir::Value methodV =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, method, 32);

    // rhs: 默认空串
    std::string rhsName = "";
    if (auto rhs = op->getAttrOfType<mlir::SymbolRefAttr>("rhs")) {
      rhsName = rhs.getRootReference().str();
    }

    // i8*（LLVM opaque ptr）
    auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(module.getContext());

    // 生成（或复用）全局字符串，传入 runtime
    // 注意：symName 要稳定且唯一；空 rhs 用固定名字
    std::string gsym = rhsName.empty() ? "_neptune_sym_time_rhs_empty"
                                       : ("_neptune_sym_time_rhs_" + rhsName);

    mlir::Value rhsPtr =
        getGlobalStringPtr(loc, module, rewriter, gsym, rhsName);

    // runtime callee: prefix + "time_advance"
    std::string nameStr = prefix + "time_advance";
    mlir::StringRef name(nameStr);

    // 关键：runtime func 签名升级为 (state, f64, i32, i8*) -> state
    auto callee = getOrCreateRuntimeFunc(
        op, name,
        /*argTypes=*/
        {stTy, rewriter.getF64Type(), rewriter.getI32Type(), i8PtrTy},
        /*resTypes=*/{outTy}, rewriter);

    auto call = rewriter.create<mlir::func::CallOp>(
        loc, callee.getSymName(), mlir::TypeRange{outTy},
        mlir::ValueRange{op.getState(), dtF64, methodV, rhsPtr});

    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }
};

} // namespace

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRRUNTIMELOWERINGPASS
#include "Passes/NeptuneIRPasses.h.inc"

struct NeptuneIRRuntimeLoweringPass final
    : public impl::NeptuneIRRuntimeLoweringPassBase<
          NeptuneIRRuntimeLoweringPass> {

  using Base =
      impl::NeptuneIRRuntimeLoweringPassBase<NeptuneIRRuntimeLoweringPass>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    if (clLinearMode == LinearLoweringMode::MatrixFree) {
      module.emitError() << "matrix-free lowering selected, but runtime ABI is "
                            "not implemented yet";
      signalPassFailure();
      return;
    }

    std::string prefix = ("_neptune_rt_" + runtime.getArgStr() + "_").str();

    RewritePatternSet patterns(ctx);

    patterns.add<AssembleMatrixLowering, SolveLinearLowering,
                 SolveNonlinearLowering, TimeAdvanceRuntimeLowering>(ctx,
                                                                     prefix);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::Neptune::NeptuneIR
