// RUN: neptune-opt %s --normalize-neptune-ir-storage --neptune-ir-stencil-to-scf | FileCheck %s

module {
  // CHECK-LABEL: func.func @jacobi_1d
  // CHECK-SAME: %[[IN:.*]]: memref<8xf64>
  // CHECK-SAME: -> memref<8xf64>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xf64>
  // CHECK: scf.for
  // CHECK:   memref.load %[[IN]]{{.*}} : memref<8xf64>
  // CHECK:   memref.store {{.*}}, %[[ALLOC]]{{.*}} : memref<8xf64>

  func.func @jacobi_1d(
      %in : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [1], ub = [9]>,
              location = #neptune_ir.location<"cell">>)
      -> !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [1], ub = [9]>,
              location = #neptune_ir.location<"cell">> {
    %out = neptune_ir.apply (%in)
      attributes { bounds = #neptune_ir.bounds<lb = [1], ub = [9]> }
      : (!neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [1], ub = [9]>,
           location = #neptune_ir.location<"cell">>)
        -> !neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [1], ub = [9]>,
           location = #neptune_ir.location<"cell">>
    {
      ^bb0(%i : index):
        %c0_25 = arith.constant 2.500000e-01 : f64

        %left = neptune_ir.access %in[-1]
          : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [1], ub = [9]>,
              location = #neptune_ir.location<"cell">> -> f64

        %center = neptune_ir.access %in[0]
          : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [1], ub = [9]>,
              location = #neptune_ir.location<"cell">> -> f64

        %right = neptune_ir.access %in[1]
          : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [1], ub = [9]>,
              location = #neptune_ir.location<"cell">> -> f64

        %sum1 = arith.addf %left, %right : f64
        %sum2 = arith.addf %sum1, %center : f64
        %new  = arith.mulf %sum2, %c0_25 : f64

        neptune_ir.yield %new : f64
    }

    return %out
      : !neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [1], ub = [9]>,
           location = #neptune_ir.location<"cell">>
  }
}
