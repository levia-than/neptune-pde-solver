// RUN: not neptune-opt %s 2>&1 | FileCheck %s

module {
  neptune_ir.linear_opdef @A : (!neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> {
    ^bb0(%arg0: !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>):
      neptune_ir.return %arg0 : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
  }
  // Static dimension should fail assemble_matrix verifier (requires ?x?xf64)
  %m = neptune_ir.assemble_matrix @A : memref<4x4xf64>
}

// CHECK: result memref must have dynamic dims
