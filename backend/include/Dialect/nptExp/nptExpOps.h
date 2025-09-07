#ifndef NPTEXPOPS_H
#define NPTEXPOPS_H

#include "mlir/IR/OpDefinition.h"
#include "nptExpDialect.h"

namespace mlir {
namespace nptExp {

// 定义一个简单的操作
class ExampleOp : public Op<ExampleOp, OpTrait::ZeroOperands, OpTrait::ZeroResult> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "nptExp.example"; }
};

} // namespace nptExp
} // namespace mlir

#endif // NPTEXPOPS_H