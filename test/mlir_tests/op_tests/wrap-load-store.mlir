// RUN: neptune-opt %s --normalize-neptune-ir-storage --neptune-ir-stencil-to-scf | FileCheck %s

module {
  // CHECK-LABEL: func.func @wrap_load_store
  // CHECK: memref.copy %{{.*}}, %{{.*}} : memref<10xf64> to memref<10xf64>

  func.func @wrap_load_store(%in : memref<10xf64>, %out : memref<10xf64>) {
    %field_in = neptune_ir.wrap %in
      : memref<10xf64>
        -> !neptune_ir.field<
             element = f64,
             bounds = #neptune_ir.bounds<lb = [0], ub = [10]>,
             location = #neptune_ir.location<"cell">>

    %temp = neptune_ir.load %field_in
      : !neptune_ir.field<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [0], ub = [10]>,
           location = #neptune_ir.location<"cell">>
        -> !neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [0], ub = [10]>,
           location = #neptune_ir.location<"cell">>

    %field_out = neptune_ir.wrap %out
      : memref<10xf64>
        -> !neptune_ir.field<
             element = f64,
             bounds = #neptune_ir.bounds<lb = [0], ub = [10]>,
             location = #neptune_ir.location<"cell">>

    neptune_ir.store %temp to %field_out
      : !neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [0], ub = [10]>,
           location = #neptune_ir.location<"cell">>
        to !neptune_ir.field<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [0], ub = [10]>,
           location = #neptune_ir.location<"cell">>

    return
  }
}
