// RUN: not neptune-opt %s 2>&1 | FileCheck %s

module {
  %c4 = arith.constant 4 : index
  %bad = memref.alloc(%c4, %c4) : memref<?x?xf32>
  %tensor = arith.constant dense<0.0> : tensor<4xf64>
  %t = neptune_ir.from_tensor %tensor : tensor<4xf64> -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
  // Wrong element type (f32) should be rejected by solve_linear verifier
  %x = neptune_ir.solve_linear %bad, %t : memref<?x?xf32>, !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
}

// CHECK: system element type must be f64
