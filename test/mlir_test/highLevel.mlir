// RUN: neptune-opt %s -verify-diagnostics 2>&1 | FileCheck %s

module {
  // CHECK-LABEL: @heat_step
  func.func @heat_step(%u_old: memref<?xf32>, %u_new: memref<?xf32>,
                  %n: index, %alpha: f32, %dt_dx2: f32) {
    // CHECK: neptune_ir.field.ref
    // CHECK: neptune_ir.field.add
    // CHECK: neptune_ir.field.sub
    // CHECK: neptune_ir.field.scale
    // CHECK: neptune_ir.evaluate
    // CHECK: neptune_ir.swap

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %n step %c1 {
      %ip1 = arith.addi %i, %c1 : index
      %im1 = arith.subi %i, %c1 : index

      // element references (symbolic)
      %fe_i   = neptune_ir.field.ref %u_old[%i] : memref<?xf32> -> !neptune_ir.field_elem<etype=f32>{}
      %fe_ip1 = neptune_ir.field.ref %u_old[%ip1] : memref<?xf32> -> !neptune_ir.field_elem<etype=f32>{}
      %fe_im1 = neptune_ir.field.ref %u_old[%im1] : memref<?xf32> -> !neptune_ir.field_elem<etype=f32>{}

      // laplacian: (u[i+1] - u[i]) + (u[i] - u[i-1]) = u[i+1] - 2*u[i] + u[i-1]
      %t1 = neptune_ir.field.sub %fe_ip1, %fe_i : !neptune_ir.field_elem<etype=f32>{}
      %t2 = neptune_ir.field.sub %fe_i,   %fe_im1 : !neptune_ir.field_elem<etype=f32>{}
      %lap = neptune_ir.field.add %t1, %t2 : !neptune_ir.field_elem<etype=f32>{}

      // scale laplacian by alpha and dt_dx2
      %s1 = neptune_ir.field.scale %lap, %alpha : !neptune_ir.field_elem<etype=f32>{}
      %s2 = neptune_ir.field.scale %s1,  %dt_dx2 : !neptune_ir.field_elem<etype=f32>{}

      // u_i + scaled laplacian
      %expr = neptune_ir.field.add %fe_i, %s2 : !neptune_ir.field_elem<etype=f32>{}

      // materialize to destination memref
      neptune_ir.evaluate %u_new, %expr : memref<?xf32>, !neptune_ir.field_elem<etype=f32>{}
    }

    // swap buffers
    neptune_ir.swap %u_old, %u_new : memref<?xf32>, memref<?xf32>

    return
  }
}