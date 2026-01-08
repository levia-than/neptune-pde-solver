#include "Frontend/NeptuneCompiler.h"

#include <stdexcept>

#include "Passes/NeptuneIRPasses.h"
#include "Passes/NeptuneIRPassesPipeline.h"

// MLIR Includes
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// LLVM Includes
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

using namespace mlir;
using namespace mlir::Neptune::NeptuneIR;

PyValue::PyValue() = default;

PyValue::PyValue(mlir::Value v) : value(v) {}

std::string PyValue::repr() {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (value)
    value.print(os);
  else
    os << "<Null Value>";
  return s;
}

NeptuneCompiler::NeptuneCompiler() {
  // 初始化 MLIR 环境
  context.getOrLoadDialect<NeptuneIRDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  // 创建 OpBuilder 和 Module
  builder = std::make_unique<mlir::OpBuilder>(&context);
  module = mlir::ModuleOp::create(builder->getUnknownLoc());

  // 初始插入点设在 Module Body 的开头
  builder->setInsertionPointToStart(module->getBody());
}

PyValue NeptuneCompiler::createWrap(PyValue buffer, std::string type_hint) {
  // MVP: 暂时忽略 type_hint，根据 buffer 创建 FieldType
  // 实际项目中你需要解析 type_hint 来创建正确的 FieldType
  // 这里假设 FieldType 已经在别处创建好了，或者 buffer 已经有类型信息
  // 为了跑通流程，我们先用 buffer 的类型做 dummy

  // TODO: 实现 createFieldType 逻辑
  // auto ftype = ...
  // auto op = builder->create<WrapOp>(loc(), ftype, buffer.value);
  // return {op.getVarField()};

  // 占位返回
  return buffer;
}

PyValue NeptuneCompiler::createAccess(PyValue temp,
                                      std::vector<int64_t> offsets) {
  auto offsetsAttr = builder->getDenseI64ArrayAttr(offsets);

  // 推导 ElementType
  auto tempType = llvm::cast<TempType>(temp.value.getType());
  mlir::Type resType = tempType.getElementType();

  auto op = builder->create<AccessOp>(loc(), resType, temp.value, offsetsAttr);
  return {op.getResult()};
}

PyValue NeptuneCompiler::createArithAdd(PyValue lhs, PyValue rhs) {
  auto op = builder->create<mlir::arith::AddFOp>(loc(), lhs.value, rhs.value);
  return {op.getResult()};
}

PyValue NeptuneCompiler::createArithSub(PyValue lhs, PyValue rhs) {
  auto op = builder->create<mlir::arith::SubFOp>(loc(), lhs.value, rhs.value);
  return {op.getResult()};
}

PyValue NeptuneCompiler::createArithMul(PyValue lhs, PyValue rhs) {
  auto op = builder->create<mlir::arith::MulFOp>(loc(), lhs.value, rhs.value);
  return {op.getResult()};
}

PyValue NeptuneCompiler::createConstant(double value) {
  // 创建 f64 类型常量
  // 对应 MLIR: %cst = arith.constant 2.0 : f64
  auto type = builder->getF64Type();
  auto attr = builder->getF64FloatAttr(value);
  auto op = builder->create<mlir::arith::ConstantOp>(loc(), type, attr);
  return {op.getResult()};
}

PyValue
NeptuneCompiler::createApply(std::vector<PyValue> inputs,
                             std::vector<int64_t> lb, std::vector<int64_t> ub,
                             py::function body_builder) { // <--- Python 回调

  // 1. 准备 Inputs
  llvm::SmallVector<mlir::Value> mlir_inputs;
  for (auto pv : inputs)
    mlir_inputs.push_back(pv.value);

  // 2. 创建 Bounds 属性
  auto boundsAttr = BoundsAttr::get(&context, builder->getDenseI64ArrayAttr(lb),
                                    builder->getDenseI64ArrayAttr(ub));

  // 3. 推导结果类型 (简化：假设结果类型 = 第一个输入的类型)
  // 实际上应该由 Python 传入或者通过推导逻辑
  auto resType = llvm::cast<TempType>(mlir_inputs[0].getType());

  // 4. 创建 ApplyOp
  auto applyOp = builder->create<ApplyOp>(loc(), resType, mlir_inputs,
                                          boundsAttr, nullptr);

  // 5. 创建 Region 和 Block
  mlir::Region &region = applyOp.getBody();
  mlir::Block *block = builder->createBlock(&region);

  // 6. 为 Block 添加参数 (TempType)
  std::vector<PyValue> block_args;
  for (auto val : mlir_inputs) {
    auto arg = block->addArgument(val.getType(), loc());
    block_args.push_back({arg});
  }

  // 7. 【关键】切换插入点 + 回调 Python
  {
    mlir::OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(block);

    // 调用 Python 函数！这时 Python 会调回 createAccess 等函数
    py::object py_res = body_builder(block_args);

    // 获取 Python 返回的结果
    PyValue res_val = py_res.cast<PyValue>();

    // 插入 Yield
    builder->create<YieldOp>(loc(), res_val.value);
  }

  return {applyOp.getResult()};
}

void NeptuneCompiler::createLinearOpDef(std::string name,
                                        std::vector<int64_t> lb,
                                        std::vector<int64_t> ub,
                                        std::string loc_kind,
                                        py::function body_builder) {

  // 1. 构造类型 (为了 Demo 简化，我们这里硬编码一个 f64 field type)
  // 实际你需要暴露 createFieldType 给 Python
  auto boundsAttr = BoundsAttr::get(&context, builder->getDenseI64ArrayAttr(lb),
                                    builder->getDenseI64ArrayAttr(ub));
  auto locAttr = Neptune::NeptuneIR::LocationAttr::get(&context, loc_kind);
  auto f64Type = builder->getF64Type();

  // 假设 Op 是 (Temp) -> (Temp)
  auto tempType = TempType::get(&context, f64Type, boundsAttr, locAttr);
  auto funcType = builder->getFunctionType({tempType}, {tempType});

  // 2. 创建 LinearOpDefOp
  auto opDef = builder->create<LinearOpDefOp>(loc(), name, funcType);

  // 3. 创建 Region/Block
  mlir::Region &region = opDef.getBody();
  mlir::Block *block = builder->createBlock(&region);

  // 添加参数
  auto arg = block->addArgument(tempType, loc());
  PyValue pyArg(arg);

  // 4. 回调 Python 填充 Body
  {
    mlir::OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(block);

    // 注意：LinearOpDef 的 body 只接受一个参数列表
    std::vector<PyValue> args = {pyArg};
    py::object py_res = body_builder(args);

    PyValue res_val = py_res.cast<PyValue>();

    // 插入 Return
    builder->create<ReturnOp>(loc(), res_val.value);
  }
  // OpDef 没有返回值，它定义了一个 Symbol
}

PyValue NeptuneCompiler::createAssembleMatrix(std::string op_symbol) {
  // 创建 SymbolRef
  auto symRef = mlir::SymbolRefAttr::get(&context, op_symbol);

  // 结果类型：memref<?x?xf64> (MVP)
  auto f64 = builder->getF64Type();
  auto memrefType = mlir::MemRefType::get(
      {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, f64);

  auto op =
      builder->create<AssembleMatrixOp>(loc(), memrefType, symRef, nullptr);
  return {op.getMatrix()};
}

PyValue NeptuneCompiler::createSolveLinear(PyValue matrix, PyValue rhs,
                                           std::string solver, double tol) {
  auto rhsType = llvm::cast<TempType>(rhs.value.getType());

  auto op = builder->create<SolveLinearOp>(loc(),
                                           rhsType, // Result type same as RHS
                                           matrix.value, rhs.value,
                                           builder->getStringAttr(solver),
                                           builder->getF64FloatAttr(tol),
                                           nullptr // max_iters optional
  );
  return {op.getResult()};
}

void NeptuneCompiler::startFunction(std::string name, std::vector<PyValue> arg_types_hints) {
  auto loc = builder->getUnknownLoc();

  // 1. 构造函数类型 (Input Types -> Result Types)
  // 暂时假设 Result Type 也是 TempType (MVP简化)
  // 实际这里应该更复杂，可能需要用户显式指定 return type
  std::vector<mlir::Type> inputTypes;
  for (auto &hint : arg_types_hints) {
    // 这里我们假设 hint 其实是一个 Type 包装器，或者是带有 Type 信息的 Value
    // 为了演示，我们暂时假设用户传进来的是 dummy value，取其 Type
    if (hint.value)
      inputTypes.push_back(hint.value.getType());
    else
      inputTypes.push_back(builder->getF64Type()); // Fallback
  }

  // 假设单返回值 (TempType)
  // TODO: 这里写死了返回类型，未来需要改成动态
  auto resultType = inputTypes.empty() ? builder->getF64Type() : inputTypes[0];
  auto funcType = builder->getFunctionType(inputTypes, {resultType});

  // 2. 创建 FuncOp
  currentFunc = builder->create<mlir::func::FuncOp>(loc, name, funcType);

  // 给函数加个 entry block
  mlir::Block *entryBlock = currentFunc.addEntryBlock();

  // 3. 将 Builder 移动到函数内部
  builder->setInsertionPointToStart(entryBlock);
}

// 结束当前函数
void NeptuneCompiler::endFunction() {
  // 将 Builder 移回 Module Body，准备迎接下一个定义
  builder->setInsertionPointAfter(currentFunc);
  currentFunc = nullptr;
}

// 获取当前函数的第 i 个参数 (作为 Value 返回给 Python 使用)
PyValue NeptuneCompiler::getFunctionArg(int index) {
  if (!currentFunc)
    throw std::runtime_error("Not inside a function!");
  mlir::Block &entryBlock = currentFunc.front();
  return {entryBlock.getArgument(index)};
}

// 创建 Return Op
void NeptuneCompiler::createReturn(PyValue retVal) {
  builder->create<mlir::func::ReturnOp>(loc(), retVal.value);
}

std::string NeptuneCompiler::dump() {
  std::string s;
  llvm::raw_string_ostream os(s);
  module->print(os);
  return s;
}

void NeptuneCompiler::runPipeline() {
  PassManager pm(&context);

  // 应用默认的 Pass Pipeline (Lowering 到 LLVM Dialect)
  // 这里的 neptunePipelineBuilder 就是你在 neptuneOpt.cpp 里定义的那个函数
  // 你需要确保它被移到了公共区域
  mlir::Neptune::NeptuneIR::buildNeptuneToLLVMPipeline(pm);

  if (failed(pm.run(module.get()))) {
    throw std::runtime_error("Failed to run MLIR Pass Pipeline");
  }
}

std::string NeptuneCompiler::compileToObjectFile(std::string output_filename) {
  // 1. 初始化 LLVM Target (必须做，否则找不到机器)
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // 2. 运行 MLIR 优化 Pipeline (Neptune -> LLVM Dialect)
  runPipeline();

  // 3. 将 MLIR 翻译为 LLVM IR (llvm::Module)
  llvm::LLVMContext llvmContext;
  mlir::registerLLVMDialectTranslation(context); // 注册翻译器
  auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);

  if (!llvmModule) {
    throw std::runtime_error("Failed to translate MLIR to LLVM IR");
  }

  // 4. 配置 TargetMachine (生成机器码的核心)
  auto tripleStr = llvm::sys::getProcessTriple();
  llvm::Triple targetTriple(tripleStr);
  llvmModule->setTargetTriple(targetTriple);

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    throw std::runtime_error("LLVM Target not found: " + error);
  }

  llvm::TargetOptions opt;
  auto rm =
      llvm::Reloc::PIC_; // 【关键】生成位置无关代码 (PIC)，因为我们要链接成 .so

  auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(targetTriple, "generic", "", opt, rm));

  llvmModule->setDataLayout(targetMachine->createDataLayout());

  // 5. 发射 .o 文件
  std::error_code ec;
  llvm::raw_fd_ostream dest(output_filename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    throw std::runtime_error("Could not open output file: " + ec.message());
  }

  llvm::legacy::PassManager pass;
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    throw std::runtime_error("TargetMachine can't emit a file of this type");
  }

  pass.run(*llvmModule);
  dest.flush();

  return output_filename;
}

mlir::Location NeptuneCompiler::loc() { return builder->getUnknownLoc(); }
