// RUN: neptune-opt %s --normalize-neptune-ir-storage --neptune-ir-stencil-to-scf | FileCheck %s

module {
  // CHECK-LABEL: func.func @laplace_2d
  // CHECK-SAME: memref<8x8xf64>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8x8xf64>
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     memref.load {{.*}} : memref<8x8xf64>
  // CHECK:     memref.store {{.*}}, %[[ALLOC]]{{.*}} : memref<8x8xf64>

  func.func @laplace_2d(
      %in : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
              location = #neptune_ir.location<"cell">>)
      -> !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
              location = #neptune_ir.location<"cell">> {
    %out = neptune_ir.apply (%in)
      attributes {
        bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
        shape  = #neptune_ir.stencil_shape<[
          [ 0,  0],
          [ 0, -1],
          [ 0,  1],
          [-1,  0],
          [ 1,  0]
        ]>
      }
      : (!neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
           location = #neptune_ir.location<"cell">>)
        -> !neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
           location = #neptune_ir.location<"cell">>
    {
      ^bb0(%i : index, %j : index):
        %c0_25 = arith.constant 2.500000e-01 : f64

        %c      = neptune_ir.access %in[0, 0]
          : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
              location = #neptune_ir.location<"cell">> -> f64
        %left   = neptune_ir.access %in[0, -1] : !neptune_ir.temp<element = f64,
              bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
              location = #neptune_ir.location<"cell">> -> f64
        %right  = neptune_ir.access %in[0, 1]  : !neptune_ir.temp<element = f64,
              bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
              location = #neptune_ir.location<"cell">> -> f64
        %down   = neptune_ir.access %in[-1, 0] : !neptune_ir.temp<element = f64,
              bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
              location = #neptune_ir.location<"cell">> -> f64
        %up     = neptune_ir.access %in[1, 0]  : !neptune_ir.temp<element = f64,
              bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
              location = #neptune_ir.location<"cell">> -> f64

        %sum1 = arith.addf %left,  %right : f64
        %sum2 = arith.addf %up,    %down  : f64
        %sum3 = arith.addf %sum1,  %sum2  : f64
        %res  = arith.mulf %sum3,  %c0_25 : f64

        neptune_ir.yield %res : f64
    }

    return %out
      : !neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [1, 1], ub = [9, 9]>,
           location = #neptune_ir.location<"cell">>
  }
}
