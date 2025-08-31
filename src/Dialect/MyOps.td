// src/Dialect/MyOps.td
include "mlir/IR/OpBase.td"
include "MyDialect.td"

// 1. 定义AddOp
def My_AddOp : Op<MyDialect, "add", [NoSideEffect]> {
  let summary = "Add two integers";
  let description = "Return the sum of two integer operands.";

  // 2. 定义操作数（两个i32类型）
  let arguments = (ins
    I32:$lhs,  // 左操作数
    I32:$rhs   // 右操作数
  );

  // 3. 定义结果（一个i32类型）
  let results = (outs I32:$result);

  // 4. 定义汇编格式（便于打印和解析）
  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `->` type($result)";

  // 5. 定义验证逻辑（可选，检查操作合法性）
  let verifier = [{
    if (getLhs().getType() != getRhs().getType()) {
      return emitOpError("operands must have the same type");
    }
    if (getResult().getType() != getLhs().getType()) {
      return emitOpError("result type must match operand type");
    }
    return success();
  }];
}
