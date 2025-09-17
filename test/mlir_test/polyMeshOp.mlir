// RUN: neptune-opt %s -verify-diagnostics 2>&1 | FileCheck %s

func.func @poly_mesh_coverage(%arg0 : i64) {
  // CHECK: npt_impl.poly_mesh(%arg0 : i64) {index = 1 : i64, isBound = true, neighbors = [2, 3], texture = 5 : i64}
  "npt_impl.poly_mesh"(%arg0) { index = 1, neighbors = [2, 3], isBound = true, texture = 5 } : (i64) -> ()
  
  // CHECK: npt_impl.poly_mesh() {index = 2 : i64}
  "npt_impl.poly_mesh"() { index = 2 } : () -> ()
  return
}
