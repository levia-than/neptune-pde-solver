#ifndef NPTEXPDIALECT_H
#define NPTEXPDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace nptImpl {

// 定义自定义方言
class NptExpDialect : public Dialect {
public:
  explicit NptExpDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "nptExp"; }
};

} // namespace nptExp
} // namespace mlir

#endif // NPTEXPDIALECT_H