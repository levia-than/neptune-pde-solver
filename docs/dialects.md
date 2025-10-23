# NeptuneIR — Dialect Reference

简短可拷贝的 Markdown 参考，描述当前工程中你实现/使用的 NeptuneIR Types & Ops、文本语法示例、最低限度的 lowering & C++ 片段，便于放到 `docs/`。

---

## 概要

NeptuneIR 是一组面向基于网格 PDE 的**中高层 IR**，用来把基于 mesh 的**字段（field）**表达为值/符号表达（`FieldElementRefType`），并最终 lowering 为嵌套循环 + `memref.load`/`arith`/`memref.store`。设计目标是把表达式层（逐元素代数）和存储层（memref/descriptor）分离，便于从 SymPy/离散化直接生成可编译 IR。

---

## Types

* **`!neptune_ir.desc` — DescriptorType**
  不透明运行时描述符（device/host 指针、stride、shape、device id 等 metadata）。

* **`!neptune_ir.field_elem` — FieldElementRefType**
  高层“元素级符号引用”，代表某个 storage 在某索引处的符号表达（未 materialize 为 scalar）。

---

## Ops（简洁列表）

> 说明格式：`name (assembly) — semantics / lowering hint`

* **`neptune_ir.module`** `(region)`
  顶层容器，module 身份。可放 storage 声明、func 等。

* **`neptune_ir.field.ref`**

  ```
  %fe = neptune_ir.field.ref %storage, %i, %j : <storage-type> -> !neptune_ir.field_elem
  ```

  返回一个元素级符号引用（storage 可以是 memref 或 descriptor，indices 为 index）。Lowering：解析 storage→memref 并把索引映射到 loop induction vars（含 halo 偏置）。

* **`neptune_ir.field.add` / `field.sub` / `field.mul` / `field.div`**

  ```
  %r = neptune_ir.field.add %lhs, %rhs : !neptune_ir.field_elem -> !neptune_ir.field_elem
  ```

  元素级二元代数运算（symbolic）。Lowering：递归展开表达式为 load/arith。

* **`neptune_ir.field.scale`**

  ```
  %r = neptune_ir.field.scale %lhs { scalar = 0.5 } : (!neptune_ir.field_elem) -> !neptune_ir.field_elem
  ```

  标量常量（FloatAttr）缩放。Lowering：在 loop 内创建与 element type 匹配的 `arith.constant` 并 `arith.mulf`。

* **`neptune_ir.evaluate`**

  ```
  neptune_ir.evaluate %dst, %expr : <dst-type>, !neptune_ir.field_elem
  ```

  把符号表达式写入目标存储（memref/descriptor）。Lowering：生成 `scf.for` 嵌套，循环体内计算表达式并 `memref.store`。

* **`neptune_ir.swap`**
  交换两个 storage。Lowering 可选择指针交换或元素拷贝（兼容 layout）。

* **`neptune_ir.desc.get` / `neptune_ir.desc.set`**
  访问 descriptor 内 metadata 的工具 op（可被 descriptor lowering 展开为 field loads/stores）。

---

## Textual assembly / parse 注意

* 如果 Type 使用自定义 mnemonic（如 `field_elem`, `desc`），必须在 dialect 的实现中包含由 TableGen 生成的类型 parser/printer (`NeptuneIROpsTypes.cpp.inc`) 并在 dialect 初始化注册 types。否则会出现 `dialect provides no type parsing hook` 错误。
* 若 assemblyFormat 中引用了 operand（如 `$expr`）且此 operand 的类型无法被 ODS 自动推断，必须在 format 中显式写 `type($expr)` 或在 op 的 args/results 中给出类型约束，否则会报：`type of operand #X is not buildable`。

---

## 示例 MLIR（heat 1D，简化，value-based）

```mlir
// RUN: mlir-opt %s -passes="lower-evaluate-to-loops" | FileCheck %s

module {
func @heat_step(%u_old: memref<128xf32>, %u_new: memref<128xf32>) {
  %i0 = arith.constant 0 : index
  %fe  = neptune_ir.field.ref %u_old, %i0 : memref<128xf32> -> !neptune_ir.field_elem
  %v   = neptune_ir.field.scale %fe { scalar = 0.5 } : (!neptune_ir.field_elem) -> !neptune_ir.field_elem
  neptune_ir.evaluate %u_new, %v : memref<128xf32>, !neptune_ir.field_elem
  return
}
}

// CHECK-LABEL: func @heat_step
// CHECK: scf.for
// CHECK: memref.load %u_old[
// CHECK: arith.mulf
// CHECK: memref.store
// CHECK-NOT: neptune_ir.evaluate
```

---

## Lowering（概要）

目标：`neptune_ir.evaluate` → `scf.for` 嵌套 + `memref.load` / `arith.*` / `memref.store`。

关键步骤：

1. 找到 `evaluate`：获取 destination storage（memref 或 descriptor），决定 loop bounds（来自 descriptor 或存储的 shape）。
2. 生成嵌套 `scf.for`（n 维），每层使用 `index` induction var。
3. 在 innermost loop 内，递归 `lowerExprToScalar(expr)`：

   * `field.ref` → `memref.load`（根据 indices 映射到 induction var / halo）
   * `field.add/mul/...` → 相应 `arith.add/mulf/...`
   * `field.scale` → 构造 `arith.constant`（注意：传入 **result type + TypedAttr**）
4. `memref.store` 结果到 dst。

实现细节提示：

* 使用 `IRMapping` 替代 `BlockAndValueMapping`。
* 在构造 `arith::ConstantOp` 时显式传入类型和 `TypedAttr`：`b.create<arith::ConstantOp>(loc, resultType, typedAttr)`（对 index 类型使用 `arith::ConstantIndexOp`）。
* 为 storage-name → memref 的解析提供 hook（`resolveStorageToMemref`），把 descriptor 或 symbol 映射到实际 memref Value / alloc。

---

## 最小 C++ 片段（可直接拷贝）

### 工厂函数（放在实现文件底部，命名空间必须一致）

```cpp
namespace mlir {
namespace Neptune {
namespace NeptuneIR {

std::unique_ptr<mlir::Pass> createLowerEvaluateToLoopPass() {
  return std::make_unique<LowerEvaluateToLoopsPass>();
}

} // namespace NeptuneIR
} // namespace Neptune
} // namespace mlir
```

### 创建浮点常量（MLIR v21 风格兼容）

```cpp
#include "llvm/ADT/APFloat.h"
static Value createFloatConstant(OpBuilder &b, Location loc, FloatType fTy, double v) {
  auto attr = FloatAttr::get(fTy, APFloat(v));
  return b.create<arith::ConstantOp>(loc, fTy, attr);
}
```

### Storage verifier stub（示例）

```cpp
LogicalResult NeptuneIR_StorageOp::verify() {
  if (!getSymName())
    return emitOpError("storage must have a name (SymbolNameAttr).");
  if (!getElementType() || !getElementType().isa<TypeAttr>())
    return emitOpError("storage must have an elementType attribute (TypeAttr).");
  if (auto shape = getShape()) {
    if (!shape.isa<ArrayAttr>()) return emitOpError("shape must be an ArrayAttr");
    for (auto a : shape.cast<ArrayAttr>())
      if (!a.isa<IntegerAttr>()) return emitOpError("shape elements must be IntegerAttr");
  }
  return success();
}
```

---

## 常见问题速查（短）

* **`dialect provides no type parsing hook`**
  -> 包含并注册由 TableGen 生成的 type parser/printer（`NeptuneIROpsTypes.cpp.inc`），并在 Dialect::initialize 中 `addTypes<...>()`。

* **`type of operand #X is not buildable`（assemblyFormat 相关）**
  -> 在 assemblyFormat 中显式写 `type($operand)` 或给 op 加类型约束。

* **`undefined reference to createLowerEvaluateToLoopPass()`（链接）**
  -> 确保头声明与实现名字/命名空间完全一致；实现文件被添加到正确的 CMake target；或使用 `PassRegistration` 静态注册替代工厂。

* **`arith::ConstantOp` build overload 不匹配**
  -> 显式使用 `b.create<arith::ConstantOp>(loc, resultType, typedAttr)` 或 `arith::ConstantIndexOp`。

* **TableGen 生成的 `.inc` 被重复包含导致 redefinition**
  -> header 只包含 `GET_OP_CLASSES` 的 `NeptuneIROps.h.inc`；cpp 包含 `GET_OP_LIST` 的 `NeptuneIROps.cpp.inc`。types 的 `GET_TYPEDEF_*` 也类似管理。

---
