module {
  neptune_ir.linear_opdef @A : (!neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> {
    ^bb0(%arg0: !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>):
      %0 = neptune_ir.apply(%arg0) attributes {bounds = #neptune_ir.bounds<lb = [0], ub = [4]>} : (!neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> {
        ^bb0(%i0: index):
          %cn2 = arith.constant -2.0 : f64
          %v0 = neptune_ir.access %arg0[0] : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> f64
          %vL = neptune_ir.access %arg0[-1] : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> f64
          %vR = neptune_ir.access %arg0[1] : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> f64
          %tmp = arith.mulf %v0, %cn2 : f64
          %tmp2 = arith.addf %tmp, %vL : f64
          %tmp3 = arith.addf %tmp2, %vR : f64
          neptune_ir.yield %tmp3 : f64
      }
      neptune_ir.return %0 : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
  }

  func.func @entry(%arg0: memref<?xf64>, %arg1: memref<?xf64>) -> memref<?xf64> {
    %f0 = neptune_ir.wrap %arg0 : memref<?xf64> -> !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    %f1 = neptune_ir.wrap %arg1 : memref<?xf64> -> !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    %t0 = neptune_ir.load %f1 : !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    %y = neptune_ir.apply_linear @A(%t0) : (!neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    neptune_ir.store %y to %f0 : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> to !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    %res = neptune_ir.unwrap %f0 : !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> memref<?xf64>
    return %res : memref<?xf64>
  }
}
