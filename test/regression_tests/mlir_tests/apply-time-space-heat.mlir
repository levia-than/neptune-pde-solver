// RUN: neptune-opt %s --normalize-neptune-ir-storage --neptune-ir-stencil-to-scf | FileCheck %s

module {
  // CHECK-LABEL: func.func @heat_time_space
  // CHECK-SAME: memref<4x6xf64>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4x6xf64>
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     memref.load {{.*}} : memref<4x6xf64>
  // CHECK:     memref.store {{.*}}, %[[ALLOC]]{{.*}} : memref<4x6xf64>

  func.func @heat_time_space(
      %u : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [0, 0], ub = [4, 6]>,
              location = #neptune_ir.location<"cell">>)
      -> !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [0, 0], ub = [4, 6]>,
              location = #neptune_ir.location<"cell">> {

    %alpha = arith.constant 1.000000e-01 : f64

    %u_next = neptune_ir.apply (%u)
      attributes { bounds = #neptune_ir.bounds<lb = [1, 1], ub = [4, 5]> }
      : (!neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [0, 0], ub = [4, 6]>,
           location = #neptune_ir.location<"cell">>)
        -> !neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [0, 0], ub = [4, 6]>,
           location = #neptune_ir.location<"cell">>
    {
      ^bb0(%t : index, %i : index):
        // 全部从 t-1 slice 读
        %u_c  = neptune_ir.access %u[-1, 0]
          : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [0, 0], ub = [4, 6]>,
              location = #neptune_ir.location<"cell">> -> f64
        %u_l  = neptune_ir.access %u[-1, -1]
          : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [0, 0], ub = [4, 6]>,
              location = #neptune_ir.location<"cell">> -> f64
        %u_r  = neptune_ir.access %u[-1, 1]
          : !neptune_ir.temp<
              element = f64,
              bounds = #neptune_ir.bounds<lb = [0, 0], ub = [4, 6]>,
              location = #neptune_ir.location<"cell">> -> f64

        %two  = arith.constant 2.000000e+00 : f64
        %twoc = arith.mulf %two, %u_c : f64

        %lap1 = arith.subf %u_l, %twoc : f64
        %lap  = arith.addf %lap1, %u_r : f64

        %alpha_lap = arith.mulf %alpha, %lap : f64
        %u_new     = arith.addf %u_c, %alpha_lap : f64

        neptune_ir.yield %u_new : f64
    }

    return %u_next
      : !neptune_ir.temp<
           element = f64,
           bounds = #neptune_ir.bounds<lb = [0, 0], ub = [4, 6]>,
           location = #neptune_ir.location<"cell">>
  }
}
