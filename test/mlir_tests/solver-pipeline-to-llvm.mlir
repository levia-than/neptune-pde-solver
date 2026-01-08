// RUN: neptune-opt %s --neptuneir-to-llvm -split-input-file | FileCheck %s --check-prefix=LLVM

// -----
module {
  neptune_ir.linear_opdef @A : (!neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> {
    ^bb0(%arg0: !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>):
      %0 = neptune_ir.apply(%arg0) attributes {bounds = #neptune_ir.bounds<lb = [0], ub = [4]>} : (!neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> {
        ^bb0(%i0: index):
          %v = neptune_ir.access %arg0[0] : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> f64
          neptune_ir.yield %v : f64
      }
      neptune_ir.return %0 : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
  }

  func.func @entry(%arg0: memref<?xf64>, %arg1: memref<?xf64>) -> memref<?xf64> {
    %f0 = neptune_ir.wrap %arg0 : memref<?xf64> -> !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    %f1 = neptune_ir.wrap %arg1 : memref<?xf64> -> !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    %t0 = neptune_ir.load %f1 : !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    %h = neptune_ir.assemble_matrix @A : memref<?x?xf64>
    %y = neptune_ir.solve_linear %h, %t0 : memref<?x?xf64>, !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    neptune_ir.store %y to %f0 : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> to !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    %res = neptune_ir.unwrap %f0 : !neptune_ir.field<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> memref<?xf64>
    return %res : memref<?xf64>
  }
}


// -----
module {
  neptune_ir.linear_opdef @A : (!neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> {
    ^bb0(%arg0: !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>):
      %0 = neptune_ir.apply(%arg0) attributes {bounds = #neptune_ir.bounds<lb = [0], ub = [4]>} : (!neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> {
        ^bb0(%i0: index):
          %v = neptune_ir.access %arg0[0] : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> f64
          neptune_ir.yield %v : f64
      }
      neptune_ir.return %0 : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
  }

  func.func @entry(%arg0: !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> {
    %y = neptune_ir.apply_linear @A(%arg0) : (!neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>) -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    %h = neptune_ir.assemble_matrix @A : memref<?x?xf64>
    %z = neptune_ir.solve_linear %h, %y {solver = "dense"} : memref<?x?xf64>, !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">> -> !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
    func.return %z : !neptune_ir.temp<element = f64, bounds = #neptune_ir.bounds<lb = [0], ub = [4]>, location = #neptune_ir.location<"cell">>
  }
}


// LLVM-LABEL: module {
// LLVM-DAG: llvm.func @_neptune_rt_runtime_assemble_matrix
// LLVM-DAG: llvm.func @_neptune_rt_runtime_solve_linear
// LLVM-DAG: llvm.func @A(
// LLVM-LABEL: llvm.func @entry
// LLVM: llvm.call @_neptune_rt_runtime_assemble_matrix
// LLVM: llvm.call @_neptune_rt_runtime_solve_linear