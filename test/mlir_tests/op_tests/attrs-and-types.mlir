// RUN: neptune-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @types
  // CHECK: %arg0: !neptune_ir.field<element = f64, bounds = <lb = [0, 0], ub = [10, 20]>, location = <"cell">>, %arg1: !neptune_ir.temp<element = f32, bounds = <lb = [1], ub = [5]>, location = <"face_x">>

  func.func @types(
      %f : !neptune_ir.field<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [0, 0], ub = [10, 20]>,
              location = #neptune_ir.location<"cell">>,
      %t : !neptune_ir.temp<
              element = f32,
              bounds = #neptune_ir.bounds<lb = [1], ub = [5]>,
              location = #neptune_ir.location<"face_x">>) {
    return
  }
}
