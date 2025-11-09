// RUN: neptune-opt %s -verify-diagnostics 2>&1 | FileCheck %s
// (A cleaned, value-based FDM time-step for 1D heat equation)
//
// This example assumes `neptune_ir` ops/types are available and focuses on a
// clear, element-wise symbolic expression (finite-difference) followed by an
// evaluate that materializes into a destination memref.
//
// FileCheck expectations are simple: we want the high-level ops to appear so
// downstream passes can lower them into loops + loads/stores.

// CHECK-LABEL: @heat_step
// CHECK: neptune_ir.field.ref
// CHECK: neptune_ir.field.sub
// CHECK: neptune_ir.field.add
// CHECK: neptune_ir.field.scale
// CHECK: neptune_ir.evaluate
// CHECK: neptune_ir.swap

module {
  func.func @heat_step(%u_old: memref<?xf32>, %u_new: memref<?xf32>, %n: index) {
    // Loop constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // For simplicity this example uses compile-time scalars in field.scale.
    // If you want runtime scalars, expose a runtime-scalar-operand variant of field.scale
    // or use a different op that accepts runtime scalar Values.
    // Here alpha and dt_dx2 are baked into attributes for clarity.
    scf.for %i = %c0 to %n step %c1 {
      // neighbor indices
      %ip1 = arith.addi %i, %c1 : index
      %im1 = arith.subi %i, %c1 : index

      // element-level symbolic references into the old buffer
      %fe_i   = neptune_ir.field.ref %u_old[%i]   : memref<?xf32> -> !neptune_ir.field_elem<etype=f32>
      %fe_ip1 = neptune_ir.field.ref %u_old[%ip1] : memref<?xf32> -> !neptune_ir.field_elem<etype=f32>
      %fe_im1 = neptune_ir.field.ref %u_old[%im1] : memref<?xf32> -> !neptune_ir.field_elem<etype=f32>

      // finite-difference Laplacian (1D): u[i+1] - 2*u[i] + u[i-1]
      // build as (u[i+1] - u[i]) + (u[i] - u[i-1])
      %t1  = neptune_ir.field.sub %fe_ip1, %fe_i  : !neptune_ir.field_elem<etype=f32>, !neptune_ir.field_elem<etype=f32> -> !neptune_ir.field_elem<etype=f32>
      %t2  = neptune_ir.field.sub %fe_i,   %fe_im1: !neptune_ir.field_elem<etype=f32>, !neptune_ir.field_elem<etype=f32> -> !neptune_ir.field_elem<etype=f32>
      %lap = neptune_ir.field.add %t1, %t2        : !neptune_ir.field_elem<etype=f32>, !neptune_ir.field_elem<etype=f32> -> !neptune_ir.field_elem<etype=f32>

      // scale laplacian: first by alpha, then by dt_dx2.
      // Here we use attribute-based scales (compile-time). Example values shown.
      %s1 = neptune_ir.field.scale %lap { scalar = -2.0 } : !neptune_ir.field_elem<etype=f32> -> !neptune_ir.field_elem<etype=f32>
      // combine with u[i+1] + u[i-1] parts accounted already via t1,t2; for clarity
      // we could also scale lap directly by a single factor if desired.
      %s2 = neptune_ir.field.scale %s1 { scalar = 0.5 } : !neptune_ir.field_elem<etype=f32> -> !neptune_ir.field_elem<etype=f32>

      // new value = u[i] + (scaled laplacian)
      %expr = neptune_ir.field.add %fe_i, %s2 : !neptune_ir.field_elem<etype=f32>, !neptune_ir.field_elem<etype=f32> -> !neptune_ir.field_elem<etype=f32>

      // materialize the expression into the destination buffer at index %i
      neptune_ir.evaluate %u_new, %expr : memref<?xf32>, !neptune_ir.field_elem<etype=f32>
    }

    // semantic swap: swap buffers for next time step
    neptune_ir.swap %u_old, %u_new : memref<?xf32>, memref<?xf32>

    return
  }
}
